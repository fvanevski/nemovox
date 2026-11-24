"""One explicit local CPU Whisper worker for a bounded batch of evidence crops."""
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
    from faster_whisper import WhisperModel
    from importlib.metadata import version
    model = WhisperModel(str(checkpoint.resolve()), device="cpu", compute_type="int8",
                         local_files_only=True)
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
    json.dump({"backend":"faster-whisper","version":version("faster-whisper"),
               "checkpoint_sha256":hashes,"device":"cpu","compute_type":"int8",
               "observations":output},sys.stdout,ensure_ascii=False,allow_nan=False)


if __name__ == "__main__":
    main()
