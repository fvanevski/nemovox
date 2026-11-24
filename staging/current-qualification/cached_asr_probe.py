"""Optional CPU-only diagnostic with an already installed local faster-whisper.

No download, installation, service start, GPU use, or production substitution.
Run only if an existing CTranslate2 checkpoint and Python environment are found.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--inputs", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--region", action="append",
                   help="Explicit region(s); default is four base regions plus expanded targets.")
    a = p.parse_args()
    required = ("model.bin", "config.json", "tokenizer.json", "preprocessor_config.json")
    if not a.checkpoint.is_dir() or not all((a.checkpoint / name).is_file() for name in required):
        p.error("Complete existing local CTranslate2 model/tokenizer/preprocessor assets required")
    if a.output.exists():
        p.error("Refusing to overwrite diagnostic output")
    from faster_whisper import WhisperModel
    import importlib.metadata
    doc = json.loads(a.inputs.read_text())
    # Explicit CPU diagnostic, not a silent production device downgrade.
    model = WhisperModel(str(a.checkpoint.resolve()), device="cpu",
                         compute_type="int8", local_files_only=True)
    results = {}
    selected = set(a.region or [
        name for name, region in doc["regions"].items()
        if region.get("variant", "base") in ("base", "expanded")])
    if not selected or not selected <= set(doc["regions"]):
        raise ValueError("Unknown independent diagnostic region")
    for name, region in doc["regions"].items():
        if name not in selected:
            continue
        wav = Path(region["wav"])
        if hashlib.sha256(wav.read_bytes()).hexdigest() != region["sha256"]:
            raise ValueError(f"Audio identity mismatch: {name}")
        t0 = time.perf_counter()
        segments, info = model.transcribe(str(wav), language=region["language"],
                                          beam_size=5, word_timestamps=True,
                                          vad_filter=False)
        rows = [{"start": s.start + region["start"], "end": s.end + region["start"],
                 "text": s.text,
                 "words": [{"word": w.word, "start": w.start + region["start"],
                            "end": w.end + region["start"]}
                           for w in (s.words or [])]} for s in segments]
        results[name] = {"audio_sha256": region["sha256"], "language": info.language,
                         "rows": rows, "elapsed_s": time.perf_counter()-t0}
    digests = {}
    for f in a.checkpoint.iterdir():
        if f.is_file():
            with f.open("rb") as stream:
                digests[str(f.relative_to(a.checkpoint))] = hashlib.file_digest(stream, "sha256").hexdigest()
    a.output.write_text(json.dumps({
        "checkpoint": str(a.checkpoint.resolve()),
        "checkpoint_files": digests,
        "faster_whisper_version": importlib.metadata.version("faster-whisper"),
        "device": "cpu", "compute_type": "int8", "results": results,
    }, ensure_ascii=False, indent=2) + "\n")
if __name__ == "__main__":
    main()
