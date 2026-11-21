#!/usr/bin/env python3
"""Work Package C: Bounded ~20-case comparison of SwiftU/NInfer vs NVIDIA Riva-Translate-4B-Instruct-v2."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

WORKSPACE = Path("/workspace/nvidia-transcribe")
if str(WORKSPACE / "src") not in sys.path:
    sys.path.insert(0, str(WORKSPACE / "src"))

from nvidia_transcribe.refinement import GlossaryConfig
from nvidia_transcribe.translation import TranslationConfig, translate_utterances

OUT_PATH = WORKSPACE / "outputs/subtitle_runs/translation_comparison.json"

BENCHMARK_CASES: list[dict[str, Any]] = [
    # --- French (10 cases) ---
    {
        "case_id": "fr_01_ordinary",
        "language": "fr",
        "category": "french_ordinary",
        "prev": [{"id": "ctx_p", "speaker_id": "SPK_01", "source_text": "Tu crois vraiment qu'on va y arriver ?"}],
        "target": {"id": "utt-fr-01", "speaker_id": "SPK_00", "source_start": 10.0, "source_end": 13.0, "source_text": "Ne t'inquiète pas, on finira bien par trouver une solution avant la tombée de la nuit."},
        "next": [{"id": "ctx_n", "speaker_id": "SPK_01", "source_text": "J'espère que tu dis vrai."}],
    },
    {
        "case_id": "fr_02_colloquial",
        "language": "fr",
        "category": "french_colloquialism",
        "prev": [{"id": "ctx_p", "speaker_id": "SPK_02", "source_text": "Il a encore filé sans prévenir personne !"}],
        "target": {"id": "utt-fr-02", "speaker_id": "SPK_03", "source_start": 20.0, "source_end": 22.8, "source_text": "Oh, laisse tomber, il n'en fait toujours qu'à sa tête celui-là !"},
        "next": [],
    },
    {
        "case_id": "fr_03_names_mcog",
        "language": "fr",
        "category": "fictional_names",
        "prev": [{"id": "ctx_p", "speaker_id": "SPK_00", "source_text": "Dépêchez-vous, les Olmèques approchent du temple."}],
        "target": {"id": "utt-fr-03", "speaker_id": "SPK_01", "source_start": 30.0, "source_end": 34.5, "source_text": "Esteban, Zia et Tao doivent rejoindre Mendoza près du Grand Condor avant Zarès !"},
        "next": [{"id": "ctx_n", "speaker_id": "SPK_00", "source_text": "Pichu veille déjà depuis la colline."}],
    },
    {
        "case_id": "fr_04_idiom_surprise",
        "language": "fr",
        "category": "french_idiom_expressive",
        "prev": [{"id": "ctx_p", "speaker_id": "SPK_01", "source_text": "Regardez ce mécanisme caché derrière la statue."}],
        "target": {"id": "utt-fr-04", "speaker_id": "SPK_00", "source_start": 40.0, "source_end": 43.0, "source_text": "Diablement efficace, n'est-il pas ? Pour sûr, c'est épatant !"},
        "next": [],
    },
    {
        "case_id": "fr_05_context_elliptical",
        "language": "fr",
        "category": "context_dependent",
        "prev": [{"id": "ctx_p", "speaker_id": "SPK_01", "source_text": "Combien de médaillons faut-il pour ouvrir cette porte ?"}],
        "target": {"id": "utt-fr-05", "speaker_id": "SPK_00", "source_start": 50.0, "source_end": 52.0, "source_text": "Il en faudrait deux, mais lui n'en a qu'un."},
        "next": [{"id": "ctx_n", "speaker_id": "SPK_01", "source_text": "Alors on a encore une chance !"}],
    },
    {
        "case_id": "fr_06_rapid_exchange",
        "language": "fr",
        "category": "rapid_exchange",
        "prev": [{"id": "ctx_p", "speaker_id": "SPK_00", "source_text": "Baisse-toi !"}],
        "target": {"id": "utt-fr-06", "speaker_id": "SPK_01", "source_start": 60.0, "source_end": 61.8, "source_text": "Quoi ? Qu'est-ce qui se passe encore ?"},
        "next": [{"id": "ctx_n", "speaker_id": "SPK_00", "source_text": "Ne respirez pas ! Tous à l'intérieur !"}],
    },
    {
        "case_id": "fr_07_overlap_turn",
        "language": "fr",
        "category": "simultaneous_overlap",
        "prev": [{"id": "ctx_p", "speaker_id": "SPK_00", "source_text": "On doit récupérer le double médaillon tout de suite !"}],
        "target": {"id": "utt-fr-07", "speaker_id": "SPK_01", "source_start": 70.5, "source_end": 73.0, "source_text": "Pas question, c'est beaucoup trop dangereux sans plan !"},
        "next": [],
    },
    {
        "case_id": "fr_08_fleurs_00",
        "language": "fr",
        "category": "french_formal",
        "prev": [],
        "target": {"id": "utt-fr-08", "speaker_id": "SPK_00", "source_start": 80.0, "source_end": 86.0, "source_text": "L’accident a eu lieu en terrain montagneux, et il semblerait que cela ait été causé par un incendie malveillant."},
        "next": [],
    },
    {
        "case_id": "fr_09_fleurs_03_mashhad",
        "language": "fr",
        "category": "proper_noun",
        "prev": [],
        "target": {"id": "utt-fr-09", "speaker_id": "SPK_00", "source_start": 90.0, "source_end": 96.0, "source_text": "Le même mois, un autre avion de ligne a fait une sortie de piste à Mashhad et a heurté un mur, tuant ainsi dix-sept personnes."},
        "next": [],
    },
    {
        "case_id": "fr_10_colloquial_grumble",
        "language": "fr",
        "category": "french_colloquialism",
        "prev": [{"id": "ctx_p", "speaker_id": "SPK_00", "source_text": "Pedro, arrête de râler et aide-moi à tirer cette caisse !"}],
        "target": {"id": "utt-fr-10", "speaker_id": "SPK_01", "source_start": 100.0, "source_end": 103.5, "source_text": "Ça va, pas la peine de crier comme ça, j'ai pas quatre bras moi !"},
        "next": [],
    },
    # --- Mandarin Chinese (10 cases) ---
    {
        "case_id": "zh_01_ordinary",
        "language": "zh",
        "category": "mandarin_ordinary",
        "prev": [{"id": "ctx_p", "speaker_id": "SPK_00", "source_text": "明天早上我们几点出发？"}],
        "target": {"id": "utt-zh-01", "speaker_id": "SPK_01", "source_start": 110.0, "source_end": 113.2, "source_text": "大概七点半吧，记得把护照和车票都带上。"},
        "next": [],
    },
    {
        "case_id": "zh_02_fleurs_01_potassium",
        "language": "zh",
        "category": "chemistry_oov_char",
        "prev": [],
        "target": {"id": "utt-zh-02", "speaker_id": "SPK_00", "source_start": 120.0, "source_end": 126.0, "source_text": "钙、钾等元素属于金属，银和金等元素当然也是金属。"},
        "next": [],
    },
    {
        "case_id": "zh_03_fleurs_02_clearance",
        "language": "zh",
        "category": "mandarin_technical_dates",
        "prev": [],
        "target": {"id": "utt-zh-03", "speaker_id": "SPK_00", "source_start": 130.0, "source_end": 136.5, "source_text": "桥下垂直净空15米。该项目于2011年8月完工，但直到2017年3月才开始通车。"},
        "next": [],
    },
    {
        "case_id": "zh_04_mixed_script",
        "language": "zh",
        "category": "cjk_latin_mixed",
        "prev": [],
        "target": {"id": "utt-zh-04", "speaker_id": "SPK_00", "source_start": 140.0, "source_end": 145.5, "source_text": "Scotturb 403 路公共汽车定期发车前往辛特拉 (Sintra)，在罗卡角停靠。"},
        "next": [],
    },
    {
        "case_id": "zh_05_idiom_chengyu",
        "language": "zh",
        "category": "mandarin_idiom",
        "prev": [{"id": "ctx_p", "speaker_id": "SPK_00", "source_text": "明明已经约定好了，他怎么临时反悔？"}],
        "target": {"id": "utt-zh-05", "speaker_id": "SPK_01", "source_start": 150.0, "source_end": 153.5, "source_text": "真是画蛇添足，结果弄巧成拙，现在全乱套了。"},
        "next": [],
    },
    {
        "case_id": "zh_06_expressive_colloquial",
        "language": "zh",
        "category": "mandarin_expressive",
        "prev": [{"id": "ctx_p", "speaker_id": "SPK_00", "source_text": "你还站在这儿发什么呆？"}],
        "target": {"id": "utt-zh-06", "speaker_id": "SPK_01", "source_start": 160.0, "source_end": 163.0, "source_text": "天哪，都火烧眉毛了，赶紧先走再说！"},
        "next": [],
    },
    {
        "case_id": "zh_07_context_pronoun",
        "language": "zh",
        "category": "context_dependent",
        "prev": [{"id": "ctx_p", "speaker_id": "SPK_00", "source_text": "那个黑衣人刚才是不是往港口那边跑了？"}],
        "target": {"id": "utt-zh-07", "speaker_id": "SPK_01", "source_start": 170.0, "source_end": 172.5, "source_text": "没错，追上去，别让他溜了！"},
        "next": [],
    },
    {
        "case_id": "zh_08_rapid_exchange",
        "language": "zh",
        "category": "rapid_exchange",
        "prev": [{"id": "ctx_p", "speaker_id": "SPK_00", "source_text": "钥匙呢？"}],
        "target": {"id": "utt-zh-08", "speaker_id": "SPK_01", "source_start": 180.0, "source_end": 181.5, "source_text": "在我这儿，接住！"},
        "next": [{"id": "ctx_n", "speaker_id": "SPK_00", "source_text": "拿到了！"}],
    },
    {
        "case_id": "zh_09_overlap",
        "language": "zh",
        "category": "simultaneous_overlap",
        "prev": [{"id": "ctx_p", "speaker_id": "SPK_00", "source_text": "这并不是告别，这是一个篇章的结束。"}],
        "target": {"id": "utt-zh-09", "speaker_id": "SPK_01", "source_start": 185.0, "source_end": 188.0, "source_text": "也是新篇章的开始！"},
        "next": [],
    },
    {
        "case_id": "zh_10_science_fleurs",
        "language": "zh",
        "category": "mandarin_technical",
        "prev": [],
        "target": {"id": "utt-zh-10", "speaker_id": "SPK_00", "source_start": 190.0, "source_end": 195.0, "source_text": "科学家们可以得出结论，暗物质对其他暗物质的影响方式与普通物质相同。"},
        "next": [],
    },
]

GLOSSARY = GlossaryConfig(
    characters=("Esteban", "Zia", "Tao", "Mendoza", "Pichu", "Pedro", "Zarès"),
    terms=("Grand Condor", "Olmecs", "double medallion", "Scotturb", "Sintra", "Cape Roca"),
)


def _load_or_init() -> dict[str, Any]:
    if OUT_PATH.is_file():
        return json.loads(OUT_PATH.read_text(encoding="utf-8"))
    return {"cases": {c["case_id"]: {**c} for c in BENCHMARK_CASES}, "runs": {}}


def _save(doc: dict[str, Any]) -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    doc = _load_or_init()

    # Translate FR cases cleanly as batch/individual preserving context
    for backend in (("riva",) if mode == "--riva-only" else ("swiftu",) if mode == "--swiftu-only" else ("riva", "swiftu")):
        t0 = time.perf_counter()
        results_by_id: dict[str, dict[str, Any]] = {}
        meta_stats: list[dict[str, Any]] = []
        for lang in ("fr", "zh"):
            sub = [c for c in BENCHMARK_CASES if c["language"] == lang]
            if backend == "riva":
                # Pass target utterances strictly to Riva (single batch so model loads once)
                utts = [{**c["target"]} for c in sub]
                out_utts, tmeta = translate_utterances(
                    utts,
                    source_language=lang,
                    config=TranslationConfig(backend="riva", target_language="en"),
                    glossary=GLOSSARY,
                    resume=False,
                )
                meta_stats.append({"lang": lang, **tmeta})
                for c, o in zip(sub, out_utts):
                    results_by_id[c["case_id"]] = {"translated_text": o["translated_text"]}
            else:
                # SwiftU uses neighboring context (prev/next) as intended!
                for c in sub:
                    seq = [*c["prev"], {**c["target"]}, *c["next"]]
                    t_c0 = time.perf_counter()
                    # Translate ONLY target via batched window by placing context before/after
                    from nvidia_transcribe.refinement import OpenAIChatTextClient
                    from nvidia_transcribe.translation import build_batched_translation_messages, parse_batched_translation_response

                    cli = OpenAIChatTextClient(base_url="http://127.0.0.1:8001/v1", model="chat", timeout_s=120.0)
                    msgs = build_batched_translation_messages(
                        target_utterances=[c["target"]],
                        context_before=c["prev"],
                        context_after=c["next"],
                        source_language=lang,
                        target_language="en",
                        style="natural",
                        glossary=GLOSSARY,
                        allow_adjacent_regrouping=False,
                    )
                    raw = cli.complete(messages=msgs, max_tokens=2048, temperature=0.0)
                    parsed = parse_batched_translation_response(raw)
                    dt_c = round(time.perf_counter() - t_c0, 4)
                    txt = parsed["translations"][0]["translated_text"] if parsed.get("translations") else ""
                    results_by_id[c["case_id"]] = {"translated_text": txt, "latency_s": dt_c}
                meta_stats.append({"lang": lang, "backend": "swiftu"})

        wall = round(time.perf_counter() - t0, 3)
        for cid, r in results_by_id.items():
            doc["cases"][cid][backend] = r
        doc["runs"][backend] = {"wall_time_s": wall, "details": meta_stats}
        _save(doc)
        print(f"Completed {backend} translation benchmark ({len(results_by_id)} cases in {wall}s) -> {OUT_PATH}")


if __name__ == "__main__":
    main()
