# Issue #14 Evidence-Fusion & Generic Event-Partition Qualification (`20261123T194055Z`)

## Complete vs Blocked Connected Components (2,008 Raw French Units)
- **Complete & Reversible (`6` components -> `8` legacy replaced by `9` event turns, `94` units):**
  - `utt-0036` (`W02b` split) -> `w02b_ev_3`, `w02b_ev_4_ord100`, `w02b_ev_5_ord101_127` (`31/31`)
  - `utt-0037` (`W02b` final) -> `w02b_ev_6_final` (`12/12`)
  - `utt-0148` + `utt-0152` (`W03b` cross-chunk merge) -> `w03b_ambrosius_ophir_saba` (`29/29`)
  - `utt-0149` (`W03b` shout) -> `w03b_esteban_tao_shout` (`2/2`)
  - `utt-0153` (`W03b` 1009s) -> `w03b_savaient_viendrions` (`5/5`)
  - `utt-0187` + `utt-0188` (`W04a` cross-boundary split!) -> `w04a_turn_esteban_question`, `w04a_turn_zia_reply` (`15/15`, bypassing displaced legacy CTC `1203.25s`/`1253.16s`)
- **Blocked Incomplete Components (`3` components):**
  - `utt-0034` (`20/26` partial, leaves `c2:70..76` unreviewed)
  - `utt-0147` + `utt-0150` (`unowned_reviewed_unit_ids` = 4 pre-v1 dropped `c11:0..3` units)
  - `utt-0154` + `utt-0156` (`utt-0154` 18/18 complete, but neighbor `utt-0156` is only 1/3 reviewed -> whole component safely blocked)
