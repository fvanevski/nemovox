"""One explicit local CUDA Whisper worker for a bounded batch of evidence crops."""
import hashlib
import json
import math
import sys
from pathlib import Path


def main():
    request = json.load(sys.stdin)
    checkpoint = Path(request["checkpoint"])
    required = ("model.bin", "config.json", "tokenizer.json", "preprocessor_config.json")
    if not checkpoint.is_dir() or not all((checkpoint / p).is_file() for p in required):
        raise ValueError("Complete local multilingual CTranslate2 assets required")
    if request.get("device")!="cuda:0" or request.get("compute_type")!="int8_float16":
        raise ValueError("Explicit CUDA challenger device and compute type required")
    import ctranslate2
    from faster_whisper import WhisperModel
    from importlib.metadata import version
    if (ctranslate2.get_cuda_device_count()<1
            or "int8_float16" not in ctranslate2.get_supported_compute_types("cuda",0)):
        raise RuntimeError("CUDA int8_float16 unavailable for the challenger")
    model = WhisperModel(str(checkpoint.resolve()), device="cuda", device_index=0,
                         compute_type="int8_float16", local_files_only=True)
    native = model.model
    def runtime_identity():
        return {"device":native.device,"device_index":list(native.device_index),
                "compute_type":native.compute_type,"model_is_loaded":native.model_is_loaded,
                "is_multilingual":native.is_multilingual}
    identity = runtime_identity()
    if identity != {"device":"cuda","device_index":[0],"compute_type":"int8_float16",
                    "model_is_loaded":True,"is_multilingual":True}:
        raise RuntimeError("Loaded challenger does not match requested CUDA model identity")
    output = []
    for crop in request["crops"]:
        path = Path(crop["path"])
        if hashlib.sha256(path.read_bytes()).hexdigest() != crop["audio_sha256"]:
            raise ValueError("Independent audio input mismatch")
        segments, info = model.transcribe(str(path), language=request["language"],
                                          beam_size=5, word_timestamps=True, vad_filter=False)
        rows = []
        for index, segment in enumerate(segments):
            a, b = segment.start + crop["start"], segment.end + crop["start"]
            if not (math.isfinite(a) and math.isfinite(b) and crop["start"] <= a < b <= crop["end"]+.01):
                raise ValueError("Invalid independent ASR segment bounds")
            rows.append({"observation_id": f"{crop['id']}:{index}", "source_start": a,
                         "source_end": min(b,crop["end"]), "source_text": segment.text.strip(),
                         "words": [{"text":w.word,"start":w.start+crop["start"],
                                    "end":w.end+crop["start"]} for w in (segment.words or [])]})
        output.append({"id":crop["id"],"audio_sha256":crop["audio_sha256"],
                       "crop":[crop["start"],crop["end"]],"rows":rows,"language":info.language})
    # Symlinked cached snapshots are legitimate. Hash actual checkpoint bytes,
    # not symlink metadata, once per bounded batch.
    hashes = {}
    for name in required:
        with (checkpoint/name).open("rb") as stream:
            hashes[name] = hashlib.file_digest(stream,"sha256").hexdigest()
    if runtime_identity()!=identity:
        raise RuntimeError("Challenger runtime identity changed during inference")
    json.dump({"backend":"faster-whisper","version":version("faster-whisper"),
               "ctranslate2_version":version("ctranslate2"),"checkpoint_sha256":hashes,
               **identity,"observations":output},sys.stdout,ensure_ascii=False,allow_nan=False)


if __name__ == "__main__":
    main()
