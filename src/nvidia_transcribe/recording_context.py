"""Compact immutable episode context and source-cited discourse hints."""
from __future__ import annotations

import hashlib
import json


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def encode(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


class RecordingContext:
    """Complete dialogue where admitted; explicitly bounded subsets otherwise.

    UTF-8 bytes are a conservative input-size proxy, not tokenizer receipts.
    The caller reserves separate headroom for prompts, schemas and completion.
    """
    def __init__(self, rows, mode, max_bytes):
        if mode not in ("english", "bilingual"):
            raise ValueError("Unsupported review context representation")
        if type(max_bytes) is not int or not 1024 <= max_bytes <= 180000:
            raise ValueError("Review context budget must be 1024..180000 UTF-8 bytes")
        self.mode = mode
        self.max_bytes = max_bytes
        self.rows = []
        for i, row in enumerate(rows):
            self.rows.append({
                "id": str(row.get("utterance_id") or row.get("id") or f"utt-{i:04d}"),
                "speaker": row.get("speaker_id") or row.get("global_speaker_id"),
                "interval": [row.get("source_start"), row.get("source_end")],
                "source": str(row.get("source_text") or ""),
                "english": str(row.get("machine_translated_text") or ""),
            })
        self.identity = digest(self.rows)

    def for_targets(self, start, end):
        bilingual = self.mode == "bilingual"
        def render(i):
            row = self.rows[i]
            return [row["id"], row["speaker"], *row["interval"],
                    *([row["source"]] if bilingual else []), row["english"]]
        records = [render(i) for i in range(len(self.rows))]
        # Reserve envelope overhead, not only the encoded row data.
        available = self.max_bytes - 512
        sizes = [len(encode(row).encode()) + 1 for row in records]
        if sum(sizes) <= available:
            selected = list(range(len(records)))
        else:
            # Keep whole observations, both preceding and following. Detailed
            # targets/local evidence are supplied separately even if omitted here.
            selected = []
            for i in sorted(range(len(records)),
                            key=lambda i: (max(start-i, i-end+1, 0), i)):
                if sizes[i] <= available:
                    selected.append(i)
                    available -= sizes[i]
            selected.sort()
        return {"representation": self.mode,
                "coverage": "complete" if len(selected) == len(records) else "bounded",
                "total_occurrences": len(records), "included_occurrences": len(selected),
                "columns": ["id", "speaker", "start", "end",
                            *(["source"] if bilingual else []), "provisional_english"],
                "dialogue": [records[i] for i in selected]}


def supported_terms(item, target, all_rows):
    """Validate literal citations; a model's accepted term is still advisory."""
    result = []
    for term in item.get("discourse_terms", []):
        if not isinstance(term, dict) or set(term) != {"source", "english", "reference_ids"}:
            continue
        source, english, refs = term["source"], term["english"], term["reference_ids"]
        if (not isinstance(source, str) or not 1 <= len(source.strip()) <= 80
                or source not in target["source_text"]
                or not isinstance(english, str) or not 1 <= len(english.strip()) <= 100
                or english.casefold() not in item["corrected_text"].casefold()
                or not isinstance(refs, list) or not 1 <= len(refs) <= 4
                or not any(ref != target["utterance_id"] for ref in refs)
                or any(not isinstance(ref, str) or ref not in all_rows
                       or source not in all_rows[ref]["source_text"] for ref in refs)):
            continue
        result.append({"source": source, "english": english,
                       "reference_ids": sorted(set(refs)), "accepted_target_id": target["utterance_id"],
                       "status": "advisory_translation_interpretation"})
    return result[:4]


def relevant_terms(ledger, targets, limit=32):
    # Never apply a global string replacement. Supply conflicting interpretations
    # as alternatives, restricted to literally present mentions.
    relevant = [term for term in ledger if any(term["source"] in t["source_text"] for t in targets)]
    # Collapse repeated equal interpretations while preserving competing senses
    # and their exact accepted occurrences. Do not let frequent repetitions evict
    # every distinct term in the current target batch.
    grouped = {}
    for term in relevant:
        key = (term["source"], term["english"].casefold())
        prior = grouped.get(key)
        ids = [*(prior["accepted_target_ids"] if prior else []), term["accepted_target_id"]]
        grouped[key] = {**term, "accepted_target_ids": list(dict.fromkeys(ids))[-8:]}
    return list(grouped.values())[-limit:]
