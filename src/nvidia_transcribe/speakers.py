"""Speaker normalization and cross-chunk speaker identity reconciliation."""

from __future__ import annotations

import heapq
import math
import tempfile
from pathlib import Path
from typing import Any, Protocol

from nvidia_transcribe.chunking import ChunkSpec
from nvidia_transcribe.media import extract_wav_segment


class SpeakerEmbedder(Protocol):
    """Optional speaker embedding backend used to reconcile non-adjacent provisional speakers."""

    name: str

    def embed_wav(self, wav_path: Path) -> list[float] | None:
        """Return an L2-normalized embedding vector for a clean single-speaker WAV clip."""


class TitaNetSpeakerEmbedder:
    """Lazy CPU wrapper around NeMo's built-in ``titanet_small`` EncDecSpeakerLabelModel."""

    name = "nemo:titanet_small:cpu"

    def __init__(self, model_name: str = "titanet_small") -> None:
        self.model_name = model_name
        self.name = f"nemo:{model_name}:cpu"
        self._model: Any = None

    def _load(self) -> Any:
        if self._model is None:
            from nemo.collections.asr.models import EncDecSpeakerLabelModel

            self._model = EncDecSpeakerLabelModel.from_pretrained(self.model_name, map_location="cpu").eval()
        return self._model

    def embed_wav(self, wav_path: Path) -> list[float] | None:
        import torch

        model = self._load()
        with torch.no_grad():
            emb = model.get_embedding(str(wav_path))
            emb = torch.nn.functional.normalize(emb.squeeze(0).float(), dim=-1)
        return [float(x) for x in emb.tolist()]


def _cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b) or not a:
        return 0.0
    return sum(x * y for x, y in zip(a, b))


def _norm_tok(text: str) -> str:
    return "".join(ch for ch in (text or "").lower() if ch.isalnum())


def normalize_chunk_speakers(
    *,
    chunk_index: int,
    units: list[dict[str, Any]],
    diarization: list[dict[str, Any]],
    speaker_mapping: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Attach explicit raw, diarization, and chunk-local speaker metadata without guessing unknowns."""
    diar_by_spk: dict[str, list[tuple[float, float]]] = {}
    for seg in diarization:
        if seg.get("valid_timing") and seg.get("start") is not None and seg.get("end") is not None:
            d_key = str(seg.get("speaker"))
            diar_by_spk.setdefault(d_key, []).append((float(seg["start"]), float(seg["end"])))

    local_profiles: dict[str, dict[str, Any]] = {}
    normalized_units: list[dict[str, Any]] = []

    for u in units:
        raw_tag_val = u.get("raw_speaker_tag")
        spk_val = u.get("speaker")
        if raw_tag_val is None and spk_val is not None and str(spk_val).strip() != "":
            s_str = str(spk_val).strip()
            raw_tag_val = s_str if s_str.startswith("<spk:") else f"<spk:{s_str}>"

        if spk_val is None or str(spk_val).strip() == "":
            local_id = f"chunk_{chunk_index:03d}:unknown"
            diar_id = None
            status = "unknown"
        else:
            tag_key = str(spk_val).strip().removeprefix("<spk:").removesuffix(">")
            local_id = f"chunk_{chunk_index:03d}:spk_{tag_key}"
            diar_raw = speaker_mapping.get(tag_key)
            if diar_raw is None and raw_tag_val is not None:
                diar_raw = speaker_mapping.get(raw_tag_val)
            diar_id = str(diar_raw) if diar_raw is not None else None
            status = "diarization_mapped" if diar_id is not None else "tag_only"

        prof = local_profiles.setdefault(
            local_id,
            {
                "local_speaker_id": local_id,
                "chunk_index": chunk_index,
                "raw_speaker_tag": raw_tag_val,
                "diarization_speaker_id": diar_id,
                "attribution_status": status,
                "diarization_intervals": diar_by_spk.get(diar_id, []) if diar_id is not None else [],
                "units": [],
                "speech_duration_s": 0.0,
            },
        )
        if u.get("valid_timing") and u.get("start") is not None and u.get("end") is not None:
            prof["speech_duration_s"] += max(0.0, float(u["end"]) - float(u["start"]))
        nu = {
            **u,
            "raw_speaker_tag": raw_tag_val,
            "diarization_speaker_id": diar_id,
            "local_speaker_id": local_id,
            "attribution_status": status,
        }
        prof["units"].append(nu)
        normalized_units.append(nu)

    return normalized_units, local_profiles


def _interval_overlap(a: tuple[float, float], b: tuple[float, float]) -> float:
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def _intervals_overlap_total(
    list_a: list[tuple[float, float]],
    list_b: list[tuple[float, float]],
    win_lo: float,
    win_hi: float,
) -> tuple[float, float]:
    """Return (intersection_s, union_s) inside [win_lo, win_hi]."""
    ca = [(max(s, win_lo), min(e, win_hi)) for s, e in list_a if e > win_lo and s < win_hi]
    cb = [(max(s, win_lo), min(e, win_hi)) for s, e in list_b if e > win_lo and s < win_hi]
    ca = [(s, e) for s, e in ca if e > s]
    cb = [(s, e) for s, e in cb if e > s]
    if not ca or not cb:
        dur_a = sum(e - s for s, e in ca)
        dur_b = sum(e - s for s, e in cb)
        return 0.0, dur_a + dur_b
    inter = sum(_interval_overlap(ia, ib) for ia in ca for ib in cb)
    union = sum(e - s for s, e in ca) + sum(e - s for s, e in cb) - inter
    return round(inter, 4), round(max(union, 1e-6), 4)


def _units_intervals(units: list[dict[str, Any]]) -> list[tuple[float, float]]:
    return [
        (float(u["start"]), float(u["end"]))
        for u in units
        if u.get("valid_timing") and u.get("start") is not None and u.get("end") is not None
    ]


def _word_overlap_score(
    units_a: list[dict[str, Any]],
    units_b: list[dict[str, Any]],
    win_lo: float,
    win_hi: float,
    tol_s: float = 0.75,
) -> tuple[float, int]:
    ua = [
        u
        for u in units_a
        if u.get("valid_timing")
        and u.get("start") is not None
        and win_lo <= 0.5 * (float(u["start"]) + float(u["end"])) <= win_hi
        and _norm_tok(u.get("text", ""))
    ]
    ub = [
        u
        for u in units_b
        if u.get("valid_timing")
        and u.get("start") is not None
        and win_lo <= 0.5 * (float(u["start"]) + float(u["end"])) <= win_hi
        and _norm_tok(u.get("text", ""))
    ]
    if not ua or not ub:
        return 0.0, 0
    matched = 0
    used_b: set[int] = set()
    for a in ua:
        ta = 0.5 * (float(a["start"]) + float(a["end"]))
        na = _norm_tok(a["text"])
        for j, b in enumerate(ub):
            if j in used_b:
                continue
            tb = 0.5 * (float(b["start"]) + float(b["end"]))
            if abs(ta - tb) <= tol_s and _norm_tok(b["text"]) == na:
                used_b.add(j)
                matched += 1
                break
    return round(matched / max(min(len(ua), len(ub)), 1), 4), matched


def _best_bipartite_assignment(
    scores: dict[tuple[str, str], float],
    left_ids: list[str],
    right_ids: list[str],
    *,
    forbidden_pairs: set[tuple[str, str]] | None = None,
) -> list[tuple[str, str, float]]:
    """Exact polynomial-time O(V^3) max-weight bipartite matching with unmatched option.

    Uses min-cost max-flow (Dijkstra with potentials) over sorted node order for
    deterministic tie handling and scales cleanly beyond 8+ speakers without
    factorial enumeration or greedy degradation.
    """
    if not left_ids or not right_ids or not scores:
        return []

    l_sorted = sorted(left_ids)
    r_sorted = sorted(right_ids)
    nl = len(l_sorted)
    nr = len(r_sorted)
    forbidden = forbidden_pairs or set()

    # Graph nodes: source=0, L=[1..nl], R_real=[1+nl..nl+nr], R_dummy=[1+nl+nr..2*nl+nr], sink=2*nl+nr+1
    # Use integer micro-weights so exact comparison has zero float drift.
    SCALE = 1_000_000
    n_nodes = 2 * nl + nr + 2
    src = 0
    sink = n_nodes - 1
    adj: list[list[list[int]]] = [[] for _ in range(n_nodes)]

    def add_edge(u: int, v: int, cap: int, cost: int) -> int:
        fwd_idx = len(adj[u])
        rev_idx = len(adj[v])
        adj[u].append([v, cap, cost, rev_idx])
        adj[v].append([u, 0, -cost, fwd_idx])
        return fwd_idx

    real_edges: list[tuple[int, str, str, float, int]] = []
    for i, lid in enumerate(l_sorted):
        u = 1 + i
        add_edge(src, u, 1, 0)
        for j, rid in enumerate(r_sorted):
            if (lid, rid) in forbidden or (rid, lid) in forbidden:
                continue
            sc = float(scores.get((lid, rid), 0.0))
            if sc > 0.0:
                v = 1 + nl + j
                cost = -int(round(sc * SCALE))
                eidx = add_edge(u, v, 1, cost)
                real_edges.append((u, lid, rid, sc, eidx))
        # Dummy unmatched option with 0 weight so low/negative net edges aren't forced
        dummy_v = 1 + nl + nr + i
        add_edge(u, dummy_v, 1, 0)
        add_edge(dummy_v, sink, 1, 0)

    for j in range(nr):
        add_edge(1 + nl + j, sink, 1, 0)

    # Initial Bellman-Ford / DAG pass for negative-cost edges (graph from src is a DAG)
    pot = [0] * n_nodes
    topo = [src] + [1 + i for i in range(nl)] + [1 + nl + j for j in range(nr)] + [1 + nl + nr + i for i in range(nl)] + [sink]
    for u in topo:
        for v, cap, cost, _ in adj[u]:
            if cap > 0 and pot[u] + cost < pot[v]:
                pot[v] = pot[u] + cost

    inf = 10**18
    for _ in range(nl):
        dist = [inf] * n_nodes
        prev_u = [-1] * n_nodes
        prev_e = [-1] * n_nodes
        dist[src] = 0
        pq: list[tuple[int, int]] = [(0, src)]
        while pq:
            d, u = heapq.heappop(pq)
            if d != dist[u]:
                continue
            for ei, (v, cap, cost, _) in enumerate(adj[u]):
                if cap <= 0:
                    continue
                nd = d + cost + pot[u] - pot[v]
                if nd < dist[v]:
                    dist[v] = nd
                    prev_u[v] = u
                    prev_e[v] = ei
                    heapq.heappush(pq, (nd, v))
        if dist[sink] >= inf:
            break
        for k in range(n_nodes):
            if dist[k] < inf:
                pot[k] += dist[k]
        # Augment 1 unit along shortest path
        v = sink
        while v != src:
            u = prev_u[v]
            ei = prev_e[v]
            rev_i = adj[u][ei][3]
            adj[u][ei][1] -= 1
            adj[v][rev_i][1] += 1
            v = u

    pairs: list[tuple[str, str, float]] = []
    for u, lid, rid, sc, eidx in real_edges:
        if adj[u][eidx][1] == 0:
            pairs.append((lid, rid, sc))
    return pairs


def reconcile_speakers(
    *,
    chunks: list[ChunkSpec],
    chunk_profiles: list[dict[str, dict[str, Any]]],
    media_path: Path | None = None,
    audio_stream_index: int = 0,
    embedder: SpeakerEmbedder | None = None,
    embedding_threshold: float = 0.65,
) -> dict[str, Any]:
    """Reconcile chunk-local speakers into global recording identities with explicit evidence.

    Rules enforced:
    - Never equate speakers solely because local numeric tags match.
    - Never merge speakers that simultaneously overlap within a chunk (cannot-link).
    - Never collapse ``unknown`` speakers into ``SPK_00``.
    - Keep unresolved / ambiguous links explicit rather than guessing.
    """
    parent: dict[str, str] = {}
    all_locals: dict[str, dict[str, Any]] = {}
    for prof_map in chunk_profiles:
        for lid, prof in prof_map.items():
            all_locals[lid] = prof
            parent[lid] = lid

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def same_chunk_collision(a: str, b: str, cannot_link: set[tuple[str, str]]) -> bool:
        ra, rb = find(a), find(b)
        members_a = [k for k in all_locals if find(k) == ra]
        members_b = [k for k in all_locals if find(k) == rb]
        for ma in members_a:
            for mb in members_b:
                if (ma, mb) in cannot_link or (mb, ma) in cannot_link:
                    return True
                # Same chunk, distinct known local speakers -> never merge two distinct speakers from one chunk
                if (
                    all_locals[ma]["chunk_index"] == all_locals[mb]["chunk_index"]
                    and ma != mb
                ):
                    return True
        return False

    # Build intra-chunk cannot-link constraints (distinct local speakers in same chunk, esp. concurrent ones)
    cannot_link: set[tuple[str, str]] = set()
    concurrent_pairs: list[dict[str, Any]] = []
    for prof_map in chunk_profiles:
        known = [lid for lid in prof_map if not lid.endswith(":unknown")]
        for i, a in enumerate(known):
            for b in known[i + 1 :]:
                cannot_link.add((a, b))
                ints_a = prof_map[a]["diarization_intervals"] or _units_intervals(prof_map[a]["units"])
                ints_b = prof_map[b]["diarization_intervals"] or _units_intervals(prof_map[b]["units"])
                ov = sum(_interval_overlap(ia, ib) for ia in ints_a for ib in ints_b)
                if ov >= 0.25:
                    concurrent_pairs.append({"speaker_a": a, "speaker_b": b, "overlap_s": round(ov, 3)})

    evidence_log: list[dict[str, Any]] = []
    ambiguous_matches: list[dict[str, Any]] = []
    collisions_prevented: list[dict[str, Any]] = []

    # Stage 1: Temporal + lexical bipartite matching across adjacent chunks' shared overlap window.
    for idx in range(len(chunks) - 1):
        win_lo = chunks[idx + 1].infer_start_s
        win_hi = chunks[idx].infer_end_s
        left_ids = [lid for lid in chunk_profiles[idx] if not lid.endswith(":unknown")]
        right_ids = [lid for lid in chunk_profiles[idx + 1] if not lid.endswith(":unknown")]
        if not left_ids or not right_ids:
            continue

        scores: dict[tuple[str, str], float] = {}
        pair_details: dict[tuple[str, str], dict[str, Any]] = {}
        for lid in left_ids:
            lp = chunk_profiles[idx][lid]
            ints_l = lp["diarization_intervals"] or _units_intervals(lp["units"])
            for rid in right_ids:
                rp = chunk_profiles[idx + 1][rid]
                ints_r = rp["diarization_intervals"] or _units_intervals(rp["units"])
                inter_s, union_s = _intervals_overlap_total(ints_l, ints_r, win_lo, win_hi)
                iou = round(inter_s / union_s, 4) if union_s > 0 else 0.0
                w_score, w_count = _word_overlap_score(lp["units"], rp["units"], win_lo, win_hi)

                # Require genuine acoustic/lexical evidence (never equate mere tag numbers).
                if (inter_s >= 0.40 and iou >= 0.20) or (w_count >= 2 and w_score >= 0.35):
                    combined = round(0.6 * iou + 0.4 * w_score, 4)
                    scores[(lid, rid)] = combined
                    pair_details[(lid, rid)] = {
                        "diar_overlap_s": inter_s,
                        "diar_iou": iou,
                        "word_overlap_score": w_score,
                        "word_matches": w_count,
                        "combined_score": combined,
                    }

        # Identify ambiguous near-ties BEFORE committing any automatic merge (row-wise AND column-wise).
        blocked_edges: set[tuple[str, str]] = set()
        ambiguity_margin = 0.12
        for lid in left_ids:
            cands = sorted(
                ((rid, scores[(lid, rid)]) for rid in right_ids if (lid, rid) in scores),
                key=lambda x: (-x[1], x[0]),
            )
            if len(cands) >= 2 and (cands[0][1] - cands[1][1]) < ambiguity_margin:
                ambiguous_matches.append(
                    {
                        "stage": "boundary_temporal",
                        "boundary_chunks": [idx, idx + 1],
                        "window": [win_lo, win_hi],
                        "source_local": lid,
                        "candidates": [{"target_local": r, "score": sc} for r, sc in cands],
                        "resolution": "left_unresolved_not_merged",
                    }
                )
                for r, sc in cands:
                    if (cands[0][1] - sc) < ambiguity_margin:
                        blocked_edges.add((lid, r))

        for rid in right_ids:
            rev_cands = sorted(
                ((lid, scores[(lid, rid)]) for lid in left_ids if (lid, rid) in scores),
                key=lambda x: (-x[1], x[0]),
            )
            if len(rev_cands) >= 2 and (rev_cands[0][1] - rev_cands[1][1]) < ambiguity_margin:
                ambiguous_matches.append(
                    {
                        "stage": "boundary_temporal",
                        "boundary_chunks": [idx, idx + 1],
                        "window": [win_lo, win_hi],
                        "target_local": rid,
                        "candidates": [{"source_local": l, "score": sc} for l, sc in rev_cands],
                        "resolution": "left_unresolved_not_merged",
                    }
                )
                for l, sc in rev_cands:
                    if (rev_cands[0][1] - sc) < ambiguity_margin:
                        blocked_edges.add((l, rid))

        filtered_scores = {k: v for k, v in scores.items() if k not in blocked_edges}
        matched = _best_bipartite_assignment(
            filtered_scores,
            left_ids,
            right_ids,
            forbidden_pairs=cannot_link,
        )
        for lid, rid, sc in matched:
            if (lid, rid) in blocked_edges:
                continue
            if same_chunk_collision(lid, rid, cannot_link):
                collisions_prevented.append(
                    {"stage": "boundary_temporal", "left": lid, "right": rid, "score": sc}
                )
                continue
            rl, rr = find(lid), find(rid)
            if rl != rr:
                parent[rr] = rl
                evidence_log.append(
                    {
                        "type": "temporal_boundary_match",
                        "boundary_chunks": [idx, idx + 1],
                        "window": [win_lo, win_hi],
                        "local_a": lid,
                        "local_b": rid,
                        **pair_details[(lid, rid)],
                    }
                )

    # Stage 2: Optional acoustic speaker embedding reconciliation for non-adjacent provisional clusters.
    embedding_diagnostics: dict[str, Any] = {"enabled": embedder is not None, "backend": getattr(embedder, "name", None)}
    if embedder is not None and media_path is not None and media_path.is_file():
        clusters: dict[str, list[str]] = {}
        for lid in sorted(all_locals):
            if lid.endswith(":unknown"):
                continue
            clusters.setdefault(find(lid), []).append(lid)

        if len(clusters) > 1:
            cluster_embs: dict[str, list[float]] = {}
            with tempfile.TemporaryDirectory(prefix="spk_emb_") as tmpdir:
                for root_id, members in clusters.items():
                    # Choose the longest clean non-overlapping speech interval >= 1.5s across cluster members.
                    best_seg: tuple[float, float] | None = None
                    best_dur = 0.0
                    for lid in members:
                        p = all_locals[lid]
                        ints = p["diarization_intervals"] or _units_intervals(p["units"])
                        other_ints = [
                            seg
                            for o_lid in all_locals
                            if all_locals[o_lid]["chunk_index"] == p["chunk_index"] and o_lid != lid
                            for seg in (all_locals[o_lid]["diarization_intervals"] or _units_intervals(all_locals[o_lid]["units"]))
                        ]
                        for s, e in ints:
                            ov_other = sum(_interval_overlap((s, e), oi) for oi in other_ints)
                            clean_dur = (e - s) - ov_other
                            if clean_dur >= 1.5 and clean_dur > best_dur and ov_other < 0.25:
                                best_dur = clean_dur
                                best_seg = (s, min(e, s + 8.0))
                    if best_seg is not None:
                        wav_out = Path(tmpdir) / f"{root_id.replace(':', '_')}.wav"
                        try:
                            extract_wav_segment(
                                media_path,
                                start_s=best_seg[0],
                                duration_s=best_seg[1] - best_seg[0],
                                out_wav=wav_out,
                                audio_stream_index=audio_stream_index,
                            )
                            vec = embedder.embed_wav(wav_out)
                            if vec is not None:
                                cluster_embs[root_id] = vec
                        except Exception as exc:
                            embedding_diagnostics.setdefault("errors", []).append(f"{root_id}: {exc}")

            # Score all non-colliding cluster pairs above threshold, then block ambiguous near-ties before linking.
            sim_matrix: dict[tuple[str, str], float] = {}
            cand_pairs: list[tuple[float, str, str]] = []
            emb_keys = sorted(cluster_embs)
            for i, ca in enumerate(emb_keys):
                for cb in emb_keys[i + 1 :]:
                    sim = round(_cosine(cluster_embs[ca], cluster_embs[cb]), 4)
                    sim_matrix[(ca, cb)] = sim
                    sim_matrix[(cb, ca)] = sim
                    if sim >= embedding_threshold:
                        cand_pairs.append((sim, ca, cb))
            cand_pairs.sort(key=lambda x: (-x[0], x[1], x[2]))

            emb_ambiguity_margin = 0.05
            ambiguous_emb_pairs: set[tuple[str, str]] = set()
            for ca in emb_keys:
                nbrs = sorted(
                    (
                        (cb, sim_matrix[(ca, cb)])
                        for cb in emb_keys
                        if cb != ca
                        and sim_matrix[(ca, cb)] >= embedding_threshold
                        and not same_chunk_collision(ca, cb, cannot_link)
                    ),
                    key=lambda x: (-x[1], x[0]),
                )
                if len(nbrs) >= 2 and (nbrs[0][1] - nbrs[1][1]) < emb_ambiguity_margin:
                    ambiguous_matches.append(
                        {
                            "stage": "embedding",
                            "source_cluster": ca,
                            "candidates": [{"target_cluster": c, "cosine": sc} for c, sc in nbrs],
                            "resolution": "left_unresolved_not_merged",
                        }
                    )
                    for c, sc in nbrs:
                        if (nbrs[0][1] - sc) < emb_ambiguity_margin:
                            ambiguous_emb_pairs.add((ca, c))
                            ambiguous_emb_pairs.add((c, ca))

            for sim, ca, cb in cand_pairs:
                if find(ca) == find(cb):
                    continue
                if (ca, cb) in ambiguous_emb_pairs:
                    continue
                if same_chunk_collision(ca, cb, cannot_link):
                    collisions_prevented.append(
                        {"stage": "embedding", "cluster_a": ca, "cluster_b": cb, "cosine": sim}
                    )
                    continue
                ra, rb = find(ca), find(cb)
                parent[rb] = ra
                evidence_log.append(
                    {
                        "type": "speaker_embedding_match",
                        "backend": embedder.name,
                        "cluster_a": ca,
                        "cluster_b": cb,
                        "cosine_similarity": sim,
                        "threshold": embedding_threshold,
                    }
                )
            embedding_diagnostics["embedded_clusters"] = len(cluster_embs)

    # Assign deterministic global speaker IDs in chronological order of first appearance.
    root_first_seen: dict[str, tuple[int, float]] = {}
    for lid, prof in all_locals.items():
        if lid.endswith(":unknown"):
            continue
        r = find(lid)
        starts = [u["start"] for u in prof["units"] if u.get("valid_timing") and u.get("start") is not None]
        t0 = min(starts) if starts else float(prof["chunk_index"] * 10000)
        key = (prof["chunk_index"], t0)
        if r not in root_first_seen or key < root_first_seen[r]:
            root_first_seen[r] = key

    sorted_roots = sorted(root_first_seen, key=lambda r: root_first_seen[r])
    root_to_global = {r: f"SPK_{i:02d}" for i, r in enumerate(sorted_roots)}

    local_to_global: dict[str, str] = {}
    unresolved: list[dict[str, Any]] = []
    global_speakers: dict[str, dict[str, Any]] = {}

    for lid in sorted(all_locals):
        prof = all_locals[lid]
        if lid.endswith(":unknown"):
            gid = f"SPK_UNK_{prof['chunk_index']:03d}"
            local_to_global[lid] = gid
            unresolved.append({"local_speaker_id": lid, "global_speaker_id": gid, "reason": "unknown_attribution"})
        else:
            gid = root_to_global[find(lid)]
            local_to_global[lid] = gid
            if len([k for k in all_locals if not k.endswith(":unknown") and find(k) == find(lid)]) == 1 and len(chunks) > 1:
                unresolved.append(
                    {
                        "local_speaker_id": lid,
                        "global_speaker_id": gid,
                        "reason": "single_chunk_only_provisional",
                    }
                )

        g_entry = global_speakers.setdefault(
            gid,
            {
                "global_speaker_id": gid,
                "local_speaker_ids": [],
                "chunk_indices": [],
                "estimated_speech_duration_s": 0.0,
                "identity_status": "unknown" if gid.startswith("SPK_UNK_") else "reconciled",
            },
        )
        g_entry["local_speaker_ids"].append(lid)
        if prof["chunk_index"] not in g_entry["chunk_indices"]:
            g_entry["chunk_indices"].append(prof["chunk_index"])
        g_entry["estimated_speech_duration_s"] = round(
            g_entry["estimated_speech_duration_s"] + prof["speech_duration_s"], 3
        )

    for g_entry in global_speakers.values():
        if g_entry["identity_status"] != "unknown" and len(chunks) > 1 and len(g_entry["chunk_indices"]) == 1:
            g_entry["identity_status"] = "provisional_single_chunk"

    raw_local_count = sum(1 for lid in all_locals if not lid.endswith(":unknown"))
    reconciled_count = sum(1 for gid in global_speakers if not gid.startswith("SPK_UNK_"))

    return {
        "local_to_global": local_to_global,
        "speakers": sorted(global_speakers.values(), key=lambda x: x["global_speaker_id"]),
        "reconciliation": {
            "raw_local_speaker_count": raw_local_count,
            "reconciled_global_speaker_count": reconciled_count,
            "unknown_speaker_count": sum(1 for gid in global_speakers if gid.startswith("SPK_UNK_")),
            "evidence": evidence_log,
            "ambiguous_matches": ambiguous_matches,
            "unresolved_speakers": unresolved,
            "collisions_prevented": collisions_prevented,
            "concurrent_speaker_pairs": concurrent_pairs,
            "embedding_reconciliation": embedding_diagnostics,
        },
    }
