"""Immutable recognized-occurrence projection for localized ASR observations."""
from __future__ import annotations
import hashlib
import json
from types import SimpleNamespace

from nvidia_transcribe.alignment import recover_chunk_alignment
from nvidia_transcribe.source_occurrences import bind_occurrence_inventory, source_observation_id
from nvidia_transcribe.occurrence_locality import assess_occurrence_locality
from nvidia_transcribe.utterances import reconstruct_speaker_utterances


def reconstruct_acoustic_observation(raw, *, audio_sha256, crop, language):
    a,b = crop
    raw_text = ((raw.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
    raw_sha = hashlib.sha256(json.dumps(raw,ensure_ascii=False,sort_keys=True,
                    separators=(",",":"),allow_nan=False).encode()).hexdigest()
    native = recover_chunk_alignment(raw_text=raw_text,ctc_payload=raw.get("ctc_timestamps"),
        source_offset_s=a,chunk_index=0,model_path=None)
    spec = SimpleNamespace(infer_start_s=a,infer_end_s=b,core_start_s=a,core_end_s=b)
    inventory = bind_occurrence_inventory(
        observation_id=source_observation_id(audio_sha256,raw_sha,spec),
        raw_text=raw_text,projected_text=raw_text,units=native["units"],
        native_observations=native["diagnostic"]["native_ctc_observations"])
    units, locality = assess_occurrence_locality(native["units"],infer_start_s=a,infer_end_s=b)
    inventory["locality"] = locality
    rows = reconstruct_speaker_utterances(units,source_language=language,chunk_windows={0:(a,b)})
    return rows,inventory
