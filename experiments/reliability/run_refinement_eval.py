#!/usr/bin/env python3
"""Phase C GPU evaluation: Localized transcript refinement (C0 vs C1 vs C2) & patch proposals."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import httpx

WORKSPACE_ROOT = Path("/workspace/nvidia-transcribe")
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))
if str(WORKSPACE_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT / "src"))

from experiments.modular.common import FRENCH_FULL_EPISODE_WAV, load_shared_corpus
from nvidia_transcribe.client import InferenceRequestConfig, NemotronVLLMClient
from nvidia_transcribe.media import extract_wav_segment
from nvidia_transcribe.metrics import _levenshtein, normalize_latin_words, normalize_zh_chars
from nvidia_transcribe.parser import parse_speaker_turns
from nvidia_transcribe.refinement import (
    PatchProposal,
    build_c2_text_probe_messages,
    evaluate_refinement_against_reference,
    parse_c2_probe_response,
    select_suspicious_utterances,
)

BASE_URL = "http://127.0.0.1:8190/v1"
SERVED_MODEL = "nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-NVFP4"
SWIFTU_BASE_URL = "http://127.0.0.1:8001/v1"
SWIFTU_MODEL = "chat"


def _query_nvidia_smi() -> dict[str, Any]:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used,memory.free", "--format=csv,noheader,nounits"],
        text=True,
    ).strip()
    used, free = [int(x.strip()) for x in out.splitlines()[0].split(",")]
    return {"used_mib": used, "free_mib": free}


def _text_chat_completion(
    http: httpx.Client,
    *,
    messages: list[dict[str, str]],
    base_url: str = BASE_URL,
    model_name: str = SERVED_MODEL,
    enable_thinking: bool = True,
    max_tokens: int = 512,
    temperature: float = 0.0,
    include_chat_template_kwargs: bool = True,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model_name,
        "messages": messages,
        "temperature": temperature,
        "top_p": 1.0,
        "max_tokens": max_tokens,
    }
    if include_chat_template_kwargs:
        payload["chat_template_kwargs"] = {"enable_thinking": enable_thinking}
    t0 = time.perf_counter()
    resp = http.post(f"{base_url}/chat/completions", json=payload)
    latency_s = round(time.perf_counter() - t0, 4)
    resp.raise_for_status()
    data = resp.json()
    return {"raw": data, "latency_s": latency_s}


def _verify_c2_edit_with_audio(*, c0_text: str, proposed_text: str, recheck_text: str, language: str) -> dict[str, Any]:
    """Accept C2a edit ONLY if independent Nemotron audio re-transcription corroborates proposed_text over c0_text."""
    norm_fn = normalize_zh_chars if language == "zh" else normalize_latin_words
    tok_c0 = norm_fn(c0_text)
    tok_prop = norm_fn(proposed_text)
    tok_rec = norm_fn(recheck_text)
    d_c0 = _levenshtein(tok_c0, tok_rec)[0]
    d_prop = _levenshtein(tok_prop, tok_rec)[0]
    corroborated = d_prop < d_c0
    return {
        "audio_corroborated": corroborated,
        "distance_recheck_to_c0": d_c0,
        "distance_recheck_to_proposed": d_prop,
        "verified_text": proposed_text if corroborated else c0_text,
    }


def _run_c1_contextual_audio(
    client: NemotronVLLMClient,
    *,
    wav_path: Path,
    language: str,
    context_terms: str,
    custom_instruction: str | None = None,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    cfg = InferenceRequestConfig(
        mode="speaker",
        source_language=language,
        context_terms=context_terms,
        custom_instruction=custom_instruction,
        capture_ctc_timestamps=False,
        require_ctc_timestamps=False,
        enable_thinking=False,
        apply_loop_guardrail=True,
    )
    raw = client.transcribe(wav_path, cfg)
    lat = round(time.perf_counter() - t0, 4)
    text = " ".join(
        t.text.strip() for t in parse_speaker_turns(str(raw["contract"]["raw_text"])) if t.text.strip()
    )
    return {"text": text, "raw": raw, "latency_s": lat}


def _run_c2b_audio_recheck(
    client: NemotronVLLMClient,
    *,
    wav_path: Path,
    language: str,
    glossary: str,
) -> dict[str, Any]:
    """Independent audio re-examination on a bounded excerpt without biasing toward the edit."""
    lang_name = "Mandarin Chinese" if language == "zh" else "French"
    instr = (
        f"Transcribe all spoken {lang_name} audio faithfully, preserving speaker turns and exact spellings "
        f"of names/terms if clearly heard ({glossary}). Do not invent speech during music."
    )
    return _run_c1_contextual_audio(
        client,
        wav_path=wav_path,
        language=language,
        context_terms=glossary,
        custom_instruction=instr,
    )


def _assemble_cases(crops_dir: Path) -> list[dict[str, Any]]:
    corpus = load_shared_corpus()
    cases: list[dict[str, Any]] = []

    # Track 1: Verified natural clips (19 cases: 6 zh_fleurs, 4 fr_fleurs, 7 fr verified overlap/neg, 2 synth overlap)
    zh_hints = {
        "zh_fleurs_01_2006": "钙, 钾, 金属",
        "zh_fleurs_05_1805": "Scotturb, Sintra, 罗卡角",
    }
    for cid, meta in sorted(corpus["mandarin_fleurs"].items()):
        cases.append(
            {
                "case_id": cid,
                "track": "verified_natural",
                "category": "code_switching" if cid == "zh_fleurs_05_1805" else "mandarin_fleurs",
                "language": "zh",
                "wav_path": meta["wav_path"],
                "reference_text": meta["raw_transcription"],
                "reference_verified": True,
                "context_terms": zh_hints.get(cid, ""),
                "prev_text": "",
                "next_text": "",
                "source_unit_ids": [f"{cid}_u0"],
                "utterance_id": cid,
                "source_start_s": 0.0,
                "source_end_s": meta["duration_s"],
            }
        )

    fr_hints = {"fr_fleurs_03_2004": "Mashhad"}
    for cid, meta in sorted(corpus["french_fleurs"].items()):
        cases.append(
            {
                "case_id": cid,
                "track": "verified_natural",
                "category": "proper_name" if cid == "fr_fleurs_03_2004" else "french_fleurs",
                "language": "fr",
                "wav_path": meta["wav_path"],
                "reference_text": meta["raw_transcription"],
                "reference_verified": True,
                "context_terms": fr_hints.get(cid, ""),
                "prev_text": "",
                "next_text": "",
                "source_unit_ids": [f"{cid}_u0"],
                "utterance_id": cid,
                "source_start_s": 0.0,
                "source_end_s": meta["duration_s"],
            }
        )

    for cid in ("O1", "O2", "O3_music_neg", "O4", "S1", "S2", "S3"):
        m = corpus["french_verified_overlap"][cid]
        ref = m.get("reference_text")
        if ref is None:
            ref = f"{m.get('speaker_a_ref', '')} {m.get('speaker_b_ref', '')}".strip()
        cases.append(
            {
                "case_id": f"ov_{cid}",
                "track": "verified_natural",
                "category": "music_negative_control" if cid == "O3_music_neg" else "overlap",
                "language": "fr",
                "wav_path": m["path"],
                "reference_text": ref,
                "reference_verified": True,
                "context_terms": "Esteban, Zia, Tao, Mendoza, Zarès",
                "prev_text": "",
                "next_text": "",
                "source_unit_ids": [f"ov_{cid}_u0"],
                "utterance_id": cid,
                "source_start_s": float(m.get("offset_s", 0.0)),
                "source_end_s": float(m.get("offset_s", 0.0)) + float(m["duration_s"]),
            }
        )

    for ov_id, lang in (("zh_2spk_overlap", "zh"), ("fr_2spk_overlap", "fr")):
        m = corpus[ov_id]
        ref = f"{m['speaker_a']['text']} {m['speaker_b']['text']}"
        cases.append(
            {
                "case_id": ov_id,
                "track": "verified_natural",
                "category": "overlap",
                "language": lang,
                "wav_path": m["wav_path"],
                "reference_text": ref,
                "reference_verified": True,
                "context_terms": "Scotturb, Sintra" if lang == "zh" else "",
                "prev_text": "",
                "next_text": "",
                "source_unit_ids": [f"{ov_id}_u0"],
                "utterance_id": ov_id,
                "source_start_s": 0.0,
                "source_end_s": m["total_duration_s"],
            }
        )

    # Track 2: Controlled ASR error-detection & false-correction suite (8 cases: 4 injected errors, 4 clean negatives)
    controlled_specs = [
        ("ctrl_err_zh_scotturb", "zh", corpus["mandarin_fleurs"]["zh_fleurs_05_1805"], "Scottube 403 路公共汽车定期发车前往辛特拉 (Sintra)，在罗卡角停靠。", "Scotturb, Sintra, 罗卡角", True),
        ("ctrl_err_zh_oov_homophone", "zh", corpus["mandarin_fleurs"]["zh_fleurs_01_2006"], "钙、甲等元素属于金属，银和金等元素当然也是金属。", "钙, 钾, 金属", True),
        ("ctrl_err_fr_mashhad", "fr", corpus["french_fleurs"]["fr_fleurs_03_2004"], "Le même mois, un autre avion de ligne a fait une sortie de piste à Machad et a heurté un mur, tuant ainsi dix-sept personnes.", "Mashhad", True),
        ("ctrl_err_fr_compound", "fr", corpus["french_fleurs"]["fr_fleurs_00_1987"], "L'accident a eu lieu en terrain montagneux, et il semblerait que cela ait été causé par un incendie mal veillant.", "malveillant", True),
        ("ctrl_neg_zh_clean_00", "zh", corpus["mandarin_fleurs"]["zh_fleurs_00_1906"], corpus["mandarin_fleurs"]["zh_fleurs_00_1906"]["raw_transcription"], "", False),
        ("ctrl_neg_zh_clean_03", "zh", corpus["mandarin_fleurs"]["zh_fleurs_03_1734"], corpus["mandarin_fleurs"]["zh_fleurs_03_1734"]["raw_transcription"], "暗物质", False),
        ("ctrl_neg_fr_clean_01", "fr", corpus["french_fleurs"]["fr_fleurs_01_1857"], corpus["french_fleurs"]["fr_fleurs_01_1857"]["raw_transcription"], "", False),
        ("ctrl_neg_fr_clean_02", "fr", corpus["french_fleurs"]["fr_fleurs_02_1913"], corpus["french_fleurs"]["fr_fleurs_02_1913"]["raw_transcription"], "", False),
    ]
    for cid, lang, meta, seed_c0, ctx, has_error in controlled_specs:
        cases.append(
            {
                "case_id": cid,
                "track": "controlled_probe",
                "category": "injected_error" if has_error else "clean_negative_control",
                "language": lang,
                "wav_path": meta["wav_path"],
                "reference_text": meta["raw_transcription"],
                "reference_verified": True,
                "preset_c0_text": seed_c0,
                "has_known_error": has_error,
                "context_terms": ctx,
                "prev_text": "",
                "next_text": "",
                "source_unit_ids": [f"{cid}_u0"],
                "utterance_id": cid,
                "source_start_s": 0.0,
                "source_end_s": meta["duration_s"],
            }
        )

    # Track 3: Real episode bounded excerpts (12 unverified cases cropped from fr_full_episode.wav)
    fr_contract = json.loads(
        (WORKSPACE_ROOT / "outputs/chunked_runs/fr_full_episode_90s/recording.contract.json").read_text(encoding="utf-8")
    )
    fr_utts = fr_contract.get("utterances") or []
    glossary_ep = ["Esteban", "Zia", "Tao", "Mendoza", "Pichu", "Athanaos", "Zarès", "Olmèques", "Cités d'Or"]
    suspicious = select_suspicious_utterances(fr_utts, glossary_terms=glossary_ep, max_candidates=24)
    by_id = {(u.get("utterance_id") or u.get("id")): idx for idx, u in enumerate(fr_utts)}

    chosen_indices: list[int] = []
    seen = set()
    # Pick diverse time spread from suspicious candidates first, then proper-noun turns
    for cand in suspicious:
        idx = by_id.get(cand["utterance_id"])
        if idx is None:
            continue
        u = fr_utts[idx]
        if u.get("source_start") is None or u.get("source_end") is None:
            continue
        # Ensure at least 15s spacing between selected episode audit excerpts for breadth
        st = float(u["source_start"])
        if any(abs(st - float(fr_utts[p]["source_start"])) < 20.0 for p in seen):
            continue
        seen.add(idx)
        chosen_indices.append(idx)
        if len(chosen_indices) >= 12:
            break
    # Top up to 12 if needed from utterances containing proper nouns
    for idx, u in enumerate(fr_utts):
        if len(chosen_indices) >= 12:
            break
        if idx in seen or u.get("source_start") is None or u.get("source_end") is None:
            continue
        txt = u.get("source_text") or ""
        if any(name.lower() in txt.lower() for name in glossary_ep):
            seen.add(idx)
            chosen_indices.append(idx)

    for idx in chosen_indices:
        u = fr_utts[idx]
        uid = str(u.get("utterance_id") or u.get("id") or f"utt_{idx}")
        us = float(u["source_start"])
        ue = float(u["source_end"])
        crop_s = max(0.0, us - 1.0)
        crop_dur = max(1.5, min(12.0, (ue - us) + 2.0))
        crop_wav = crops_dir / f"ep_{uid}.wav"
        extract_wav_segment(FRENCH_FULL_EPISODE_WAV, out_wav=crop_wav, start_s=crop_s, duration_s=crop_dur)
        prev_txt = str(fr_utts[idx - 1].get("source_text") or "") if idx > 0 else ""
        next_txt = str(fr_utts[idx + 1].get("source_text") or "") if idx + 1 < len(fr_utts) else ""
        cases.append(
            {
                "case_id": f"ep_{uid}",
                "track": "episode_unverified_audit",
                "category": "episode_localized_excerpt",
                "language": "fr",
                "wav_path": str(crop_wav),
                "reference_text": None,
                "reference_verified": False,
                "preset_c0_text": str(u.get("source_text") or ""),
                "context_terms": ", ".join(glossary_ep),
                "prev_text": prev_txt,
                "next_text": next_txt,
                "source_unit_ids": list(u.get("source_unit_ids") or [uid]),
                "utterance_id": uid,
                "source_start_s": us,
                "source_end_s": ue,
            }
        )

    return cases


def main() -> None:
    out_dir = WORKSPACE_ROOT / "outputs/reliability_runs"
    crops_dir = out_dir / "refinement_crops"
    crops_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "phase_c_refinement.json"

    c2a_swiftu_mode = "--c2a-swiftu" in sys.argv
    c2b_audio_mode = "--c2b-audio" in sys.argv
    update_c2_mode = "--update-c2" in sys.argv or c2a_swiftu_mode or c2b_audio_mode

    client = NemotronVLLMClient(base_url=BASE_URL, model=SERVED_MODEL, timeout_s=180.0)
    http = httpx.Client(headers={"Authorization": f"Bearer {client.api_key}"}, timeout=180.0)
    http_swiftu = httpx.Client(timeout=180.0)
    vram_start = _query_nvidia_smi()

    existing_by_id: dict[str, dict[str, Any]] = {}
    prev_doc: dict[str, Any] = {}
    if update_c2_mode and out_path.is_file():
        prev_doc = json.loads(out_path.read_text(encoding="utf-8"))
        existing_by_id = {c["case_id"]: c for c in (prev_doc.get("cases") or [])}

    # 1. Explicit C2 text-only reasoning probe gate
    probe_msgs = build_c2_text_probe_messages(
        target_text="Estebane doit retrouver Mendoza.",
        language="fr",
        prev_text="Dépêchons-nous!",
        next_text="Il nous attend près du port.",
        glossary_terms=["Esteban", "Mendoza"],
    )
    c2_probe_result: dict[str, Any] = prev_doc.get("c2_reasoning_probe_gate") or {}
    c2_swiftu_probe: dict[str, Any] = prev_doc.get("c2_swiftu_reasoning_probe") or {}
    c2_enabled = False
    c2_backend = "none"

    if c2b_audio_mode:
        c2_enabled = bool(prev_doc.get("c2_enabled"))
        c2_backend = str(prev_doc.get("c2_backend") or "swiftu_ninfer")
    elif c2a_swiftu_mode:
        try:
            sr = _text_chat_completion(
                http_swiftu,
                messages=probe_msgs,
                base_url=SWIFTU_BASE_URL,
                model_name=SWIFTU_MODEL,
                max_tokens=4096,
                include_chat_template_kwargs=False,
            )
            sp = parse_c2_probe_response(sr["raw"], fallback_text="Estebane doit retrouver Mendoza.")
            c2_enabled = sp["status"] == "ok" and bool(sp.get("reasoning_content"))
            c2_backend = "swiftu_ninfer"
            c2_swiftu_probe = {
                "supported": c2_enabled,
                "model_path": "/workspace/ninfer/swiftu (Qwen3.8-27B-swift15-nvfp4)",
                "latency_s": sr["latency_s"],
                "parsed": sp,
            }
        except Exception as exc:
            c2_swiftu_probe = {"supported": False, "error": f"{type(exc).__name__}: {exc}"}
    else:
        try:
            probe_resp = _text_chat_completion(http, messages=probe_msgs, enable_thinking=True, max_tokens=512)
            parsed_probe = parse_c2_probe_response(probe_resp["raw"], fallback_text="Estebane doit retrouver Mendoza.")
            probe_resp_nt = _text_chat_completion(http, messages=probe_msgs, enable_thinking=False, max_tokens=512)
            parsed_probe_nt = parse_c2_probe_response(probe_resp_nt["raw"], fallback_text="Estebane doit retrouver Mendoza.")
            c2_enabled = parsed_probe["status"] == "ok" and bool(parsed_probe["reasoning_content"])
            c2_backend = "nemotron_vllm" if c2_enabled else "none"
            c2_probe_result = {
                "supported": c2_enabled,
                "latency_s": probe_resp["latency_s"],
                "parsed": parsed_probe,
                "parsed_no_thinking": parsed_probe_nt,
                "raw_message": (probe_resp["raw"].get("choices") or [{}])[0].get("message"),
            }
        except Exception as exc:
            c2_probe_result = {"supported": False, "error": f"{type(exc).__name__}: {exc}"}

    cases = _assemble_cases(crops_dir)
    case_results: list[dict[str, Any]] = []
    proposals: list[dict[str, Any]] = []

    for c in cases:
        cid = c["case_id"]
        lang = c["language"]
        wav_path = Path(c["wav_path"])
        prev_rec = existing_by_id.get(cid)

        # C0 baseline & C1 contextual audio (reuse cached results under --update-c2)
        if prev_rec is not None:
            c0_text = prev_rec["c0_text"]
            c0_lat = prev_rec["c0_latency_s"]
            c1_res = {"text": prev_rec["c1_text"], "latency_s": prev_rec["c1_latency_s"]}
        else:
            if "preset_c0_text" in c:
                c0_text = c["preset_c0_text"]
                c0_lat = 0.0
            else:
                t0 = time.perf_counter()
                raw0 = client.transcribe(
                    wav_path,
                    InferenceRequestConfig(
                        mode="speaker",
                        source_language=lang,
                        capture_ctc_timestamps=False,
                        require_ctc_timestamps=False,
                        enable_thinking=False,
                    ),
                )
                c0_lat = round(time.perf_counter() - t0, 4)
                c0_text = " ".join(
                    t.text.strip() for t in parse_speaker_turns(str(raw0["contract"]["raw_text"])) if t.text.strip()
                )

            lang_name = "Mandarin Chinese" if lang == "zh" else "French"
            c1_instr = (
                f"Retranscribe this {lang_name} excerpt faithfully. Pay careful attention to proper names "
                f"and rare terms ({c['context_terms'] or 'none'}). Preserve original language verbatim."
            )
            c1_res = _run_c1_contextual_audio(
                client,
                wav_path=wav_path,
                language=lang,
                context_terms=c["context_terms"],
                custom_instruction=c1_instr,
            )
        c1_prop = PatchProposal(
            proposal_id=f"{cid}_c1",
            case_id=cid,
            language=lang,
            method="C1_contextual_audio",
            source_unit_ids=c["source_unit_ids"],
            utterance_id=c["utterance_id"],
            source_start_s=c["source_start_s"],
            source_end_s=c["source_end_s"],
            before_text=c0_text,
            proposed_text=c1_res["text"],
            changed=c1_res["text"].strip() != c0_text.strip(),
            audio_verified=True,
            reasoning_content=None,
            rationale="Contextual audio retranscription",
            status="proposed_only",
        )
        proposals.append(c1_prop.to_dict())

        # C2 text-only reasoning probe + independent audio recheck if changed
        c2a_data: dict[str, Any] = {"status": "skipped_probe_unsupported", "changed": False, "proposed_text": c0_text, "latency_s": 0.0}
        c2b_data: dict[str, Any] = {"ran": False, "verified_text": c0_text, "latency_s": 0.0}
        if c2b_audio_mode and prev_rec is not None:
            c2a_data = dict(prev_rec.get("c2a") or c2a_data)
        elif c2_enabled:
            glossary_list = [x.strip() for x in c["context_terms"].split(",") if x.strip()]
            msgs = build_c2_text_probe_messages(
                target_text=c0_text,
                language=lang,
                prev_text=c["prev_text"],
                next_text=c["next_text"],
                glossary_terms=glossary_list,
            )
            try:
                if c2a_swiftu_mode:
                    r2 = _text_chat_completion(
                        http_swiftu,
                        messages=msgs,
                        base_url=SWIFTU_BASE_URL,
                        model_name=SWIFTU_MODEL,
                        max_tokens=4096,
                        include_chat_template_kwargs=False,
                    )
                else:
                    r2 = _text_chat_completion(http, messages=msgs, enable_thinking=True, max_tokens=512)
                p2 = parse_c2_probe_response(r2["raw"], fallback_text=c0_text)
                c2a_data = {**p2, "latency_s": r2["latency_s"], "backend": c2_backend}
            except Exception as exc:
                c2a_data = {"status": "error", "error": str(exc), "changed": False, "proposed_text": c0_text, "latency_s": 0.0}

        if c2_enabled or c2a_data.get("status") == "ok":
            proposals.append(
                PatchProposal(
                    proposal_id=f"{cid}_c2a",
                    case_id=cid,
                    language=lang,
                    method="C2_text_reasoning_only",
                    source_unit_ids=c["source_unit_ids"],
                    utterance_id=c["utterance_id"],
                    source_start_s=c["source_start_s"],
                    source_end_s=c["source_end_s"],
                    before_text=c0_text,
                    proposed_text=c2a_data["proposed_text"],
                    changed=bool(c2a_data.get("changed")),
                    audio_verified=False,
                    reasoning_content=c2a_data.get("reasoning_content"),
                    rationale=str(c2a_data.get("rationale") or ""),
                    status="proposed_only",
                ).to_dict()
            )

            if c2a_data.get("changed") and not c2a_swiftu_mode:
                recheck = _run_c2b_audio_recheck(client, wav_path=wav_path, language=lang, glossary=c["context_terms"])
                vcheck = _verify_c2_edit_with_audio(
                    c0_text=c0_text,
                    proposed_text=c2a_data["proposed_text"],
                    recheck_text=recheck["text"],
                    language=lang,
                )
                c2b_data = {
                    "ran": True,
                    "recheck_text": recheck["text"],
                    "latency_s": recheck["latency_s"],
                    **vcheck,
                }
                proposals.append(
                    PatchProposal(
                        proposal_id=f"{cid}_c2b",
                        case_id=cid,
                        language=lang,
                        method="C2_audio_verified",
                        source_unit_ids=c["source_unit_ids"],
                        utterance_id=c["utterance_id"],
                        source_start_s=c["source_start_s"],
                        source_end_s=c["source_end_s"],
                        before_text=c0_text,
                        proposed_text=vcheck["verified_text"],
                        changed=vcheck["audio_corroborated"],
                        audio_verified=True,
                        reasoning_content=c2a_data.get("reasoning_content"),
                        rationale=f"Audio verification (d_c0={vcheck['distance_recheck_to_c0']}, d_prop={vcheck['distance_recheck_to_proposed']})",
                        status="proposed_only",
                    ).to_dict()
                )
            elif c2a_data.get("changed") and c2a_swiftu_mode:
                c2b_data = {"ran": False, "pending_audio_recheck": True, "verified_text": c0_text, "latency_s": 0.0}

        # Evaluate metrics against verified reference where present
        eval_c0 = evaluate_refinement_against_reference(
            before_text=c0_text, proposed_text=c0_text, reference_text=c["reference_text"], language=lang
        )
        eval_c1 = evaluate_refinement_against_reference(
            before_text=c0_text, proposed_text=c1_res["text"], reference_text=c["reference_text"], language=lang
        )
        eval_c2a = evaluate_refinement_against_reference(
            before_text=c0_text, proposed_text=c2a_data["proposed_text"], reference_text=c["reference_text"], language=lang
        )
        eval_c2b = evaluate_refinement_against_reference(
            before_text=c0_text,
            proposed_text=c2b_data["verified_text"] if c2b_data["ran"] else c0_text,
            reference_text=c["reference_text"],
            language=lang,
        )

        case_results.append(
            {
                "case_id": cid,
                "track": c["track"],
                "category": c["category"],
                "language": lang,
                "reference_verified": c["reference_verified"],
                "reference_text": c["reference_text"],
                "c0_text": c0_text,
                "c0_latency_s": c0_lat,
                "c1_text": c1_res["text"],
                "c1_latency_s": c1_res["latency_s"],
                "c2a": c2a_data,
                "c2b": c2b_data,
                "metrics": {"c0": eval_c0, "c1": eval_c1, "c2a": eval_c2a, "c2b": eval_c2b},
            }
        )

    # Aggregate summary by track & method
    def _agg(track_name: str, method_key: str) -> dict[str, Any]:
        sub = [r for r in case_results if r["track"] == track_name]
        verified = [r for r in sub if r["reference_verified"]]
        outcomes: dict[str, int] = {}
        proposed_changes = 0
        true_improvements = 0
        false_corrs = 0
        initially_wrong = 0
        repaired_edits = 0
        before_vals = []
        after_vals = []
        for r in verified:
            m = r["metrics"][method_key]
            outc = m["outcome"]
            outcomes[outc] = outcomes.get(outc, 0) + 1
            if m["changed"]:
                proposed_changes += 1
            if outc == "improved":
                true_improvements += 1
            if m["false_correction"]:
                false_corrs += 1
            if not m["initially_exact"]:
                initially_wrong += 1
            repaired_edits += int(m["net_edits_reduced"])
            before_vals.append(float(m["before_metric"]))
            after_vals.append(float(m["after_metric"]))
        clean_count = sum(1 for r in verified if r["metrics"][method_key]["initially_exact"])
        return {
            "total_cases": len(sub),
            "verified_cases": len(verified),
            "proposed_changes": proposed_changes,
            "true_improvements": true_improvements,
            "false_corrections": false_corrs,
            "precision": round(true_improvements / proposed_changes, 4) if proposed_changes else None,
            "recall": round(true_improvements / initially_wrong, 4) if initially_wrong else None,
            "false_correction_rate": round(false_corrs / clean_count, 4) if clean_count else None,
            "net_edits_reduced": repaired_edits,
            "mean_metric_before": round(sum(before_vals) / len(before_vals), 4) if before_vals else None,
            "mean_metric_after": round(sum(after_vals) / len(after_vals), 4) if after_vals else None,
            "outcomes": outcomes,
        }

    vram_end = _query_nvidia_smi()
    report = {
        "base_url": BASE_URL,
        "served_model": SERVED_MODEL,
        "vram_start": vram_start,
        "vram_end": vram_end,
        "c2_enabled": c2_enabled,
        "c2_backend": c2_backend,
        "c2_reasoning_probe_gate": c2_probe_result,
        "c2_swiftu_reasoning_probe": c2_swiftu_probe,
        "total_cases": len(case_results),
        "summary": {
            "verified_natural": {m: _agg("verified_natural", m) for m in ("c0", "c1", "c2a", "c2b")},
            "controlled_probe": {m: _agg("controlled_probe", m) for m in ("c0", "c1", "c2a", "c2b")},
            "c2_quality_diagnostics": {
                "c2a_ok_count": sum(1 for r in case_results if r["c2a"].get("status") == "ok"),
                "c2a_parse_error_count": sum(1 for r in case_results if r["c2a"].get("status") == "parse_error"),
                "c2a_degenerate_repetition_count": sum(1 for r in case_results if r["c2a"].get("degenerate_repetition")),
                "c2a_flag_contradiction_count": sum(1 for r in case_results if r["c2a"].get("flag_contradiction")),
                "c2a_changed_count": sum(1 for r in case_results if r["c2a"].get("changed")),
            },
            "episode_unverified_audit": {
                "total_cases": sum(1 for r in case_results if r["track"] == "episode_unverified_audit"),
                "c1_changed": sum(
                    1
                    for r in case_results
                    if r["track"] == "episode_unverified_audit" and r["metrics"]["c1"]["changed"]
                ),
                "c2a_changed": sum(
                    1
                    for r in case_results
                    if r["track"] == "episode_unverified_audit" and r["metrics"]["c2a"]["changed"]
                ),
                "c2b_ran": sum(
                    1 for r in case_results if r["track"] == "episode_unverified_audit" and r["c2b"]["ran"]
                ),
            },
        },
        "authoritative_transcript_mutated": False,
        "cases": case_results,
        "patch_proposals": proposals,
    }
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote Phase C report ({len(case_results)} cases, c2_enabled={c2_enabled}) to {out_path}")


if __name__ == "__main__":
    main()
