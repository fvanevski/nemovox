"""Synthetic CPU reproduction of a short distinct-speaker overlap truncating a full cue."""

import json
from pathlib import Path
import tempfile

from nvidia_transcribe.subtitles import render_subtitles


def seconds(timestamp):
    hours, minutes, seconds_text = timestamp.replace(",", ".").split(":")
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds_text)


def main():
    full_line = "The courier delivered the supplies to the shelter before night."
    utterances = [
        {"id": "a", "speaker_id": "SPK_A", "source_start": 10.0,
         "source_end": 12.82, "source_text": full_line, "translated_text": full_line},
        {"id": "b", "speaker_id": "SPK_B", "source_start": 10.4,
         "source_end": 10.615, "source_text": "Yes.", "translated_text": "Yes."},
    ]
    with tempfile.TemporaryDirectory() as output:
        result = render_subtitles({"utterances": utterances}, output_dir=Path(output),
                                  episode_stem="synthetic-overlap", formats=("srt", "ass"))
        first_srt = Path(result["outputs"]["srt"]).read_text().splitlines()[:4]
        ass_dialogues = [line for line in Path(result["outputs"]["ass"]).read_text().splitlines()
                         if line.startswith("Dialogue: ")]
    start, end = map(seconds, first_srt[1].split(" --> "))
    ass_parts = ass_dialogues[0].split(",", 9)
    ass_duration = seconds(ass_parts[2]) - seconds(ass_parts[1])
    srt_duration = end - start
    print(json.dumps({
        "input_voice_a_seconds": 2.82, "input_voice_b_overlap_seconds": 0.215,
        "srt_voice_a_seconds": round(srt_duration, 3),
        "ass_voice_a_seconds": round(ass_duration, 3),
        "voice_a_characters": len(full_line),
        "srt_voice_a_characters_per_second": round(len(full_line) / srt_duration, 2),
        "ass_layers": [line.split(",", 1)[0].split()[-1] for line in ass_dialogues],
        "conflicts": result["conflicts"],
    }, indent=2))
    return int(min(srt_duration, ass_duration) < len(full_line) / 25)


if __name__ == "__main__":
    raise SystemExit(main())
