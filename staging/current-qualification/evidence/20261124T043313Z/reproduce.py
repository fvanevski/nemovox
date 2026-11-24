"""Reproduce two CPU gate failures using only synthetic inputs."""

import json
import os
from pathlib import Path
import tempfile
from unittest.mock import patch

from nvidia_transcribe import cli, source_quality as quality, subtitle_pipeline


def rerender_probe(root):
    media = root / "synthetic.wav"
    media.touch()
    with (
        patch.dict(os.environ, {}, clear=True),
        patch.object(subtitle_pipeline, "managed_asr_client",
                     side_effect=RuntimeError("CPU probe intercepted ASR acquisition")) as acquire,
    ):
        exit_code = cli.main([
            "generate-subtitles", str(media), "--output-dir", str(root / "missing"),
            "--source-language", "fr", "--re-render",
        ])
    return {"cli_exit": exit_code, "managed_asr_calls": acquire.call_count}


def source_probe():
    text = "On y va !"
    earlier = {
        "id": "original", "source_text": text, "source_start": 10.0,
        "source_end": 12.0, "speaker_id": "SPK_A", "chunk_indices": [1],
    }
    between = {
        "id": "between", "source_text": "Attends !", "source_start": 11.3,
        "source_end": 11.6, "speaker_id": "SPK_B", "chunk_indices": [1],
    }

    def observation(name, crop, start, end):
        return {
            "crop": crop, "audio_sha256": name + "-audio", "raw_response_sha256": name + "-raw",
            "utterances": [{
                "observation_id": name + ":0", "source_text": text,
                "source_start": start, "source_end": end, "local_voice": "SPK_0",
            }],
        }

    first = observation("A", [7, 18], 11.8, 13.8)
    second = observation("B", [5, 16], 11.9, 13.9)
    policy = quality.SourceQualityPolicy()
    events = quality.corroborated_events(first, second, [10, 15], policy.timing_tolerance_s)
    assert len(events) == 1, events
    events[0]["chunks"] = [1]
    recording = {
        "source": {"fingerprint_sha256": "synthetic"}, "source_language": "fr",
        "utterances": [earlier, between],
    }
    rows, decisions = quality.select_source_events(recording, events, policy)
    return {
        "earlier": earlier, "between": between, "challenge": events[0],
        "second_bounds": [11.9, 13.9], "selected_ids": [row["id"] for row in rows],
        "decisions": decisions,
    }


def main():
    with tempfile.TemporaryDirectory() as directory:
        rerender = rerender_probe(Path(directory))
    source = source_probe()
    print(json.dumps({"missing_rerender": rerender, "displaced_source": source},
                     indent=2, ensure_ascii=False))
    errors = []
    if rerender["cli_exit"] == 0 or rerender["managed_asr_calls"]:
        errors.append("Missing translated contract must reject before model acquisition.")
    if any(item["status"] == "already_represented" for item in source["decisions"]):
        errors.append("Chronologically displaced speech must remain explicit as an alternative.")
    for error in errors:
        print("FAIL:", error)
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
