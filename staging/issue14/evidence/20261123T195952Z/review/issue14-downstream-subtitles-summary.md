# Issue #14 Downstream Subtitle Qualification (`20261123T195952Z`)

- **Outcome:** `PARTIAL` due to real upstream/downstream ID-keying bug (`utterances.py:660-661` renumbering `utt-0148..0152` vs `source_fusion_downstream.py:132,200` joining on post-reorder `u["id"]`).
- **Riva 9-Turn Translation (`riva_local`):** Executed cleanly (`16.065s`, `peak 8017.14 MiB`, VRAM restored `3248 MiB`). Fixed W02b turn splits (`Me, of course!` / `You?` / `Perfect!...`), W03b untimed Ophir/Saba merge (`unrenderable 3 -> 2`), and W04a boundary recovery (`1231.20-1232.98` & `1233.00-1234.78`), while flagging literal domain loanword issue (`nacale -> native`, `langage mu -> Mute language`).
- **`c10:75..78` vs `c11:0..3`:** Proposed as alternate observation of `w03a_bateau_volant` without auto-deletion (`[980, 984]` not cropped).
