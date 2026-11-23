# Issue #14 Default Source Recovery — 22-Case Chronology Matrix

| Case | Status | Summary |
|---|---|---|
| W01a | PARTIAL | Displaced theme token 'Tincio' still appears at 80.1s/82.7s in c00; whisper large-v3 omits theme song lyrics. |
| W01b | PARTIAL | Early recap drift in c00; Whisper v3 matches clean lexical recap. |
| W01c | PARTIAL | Shouted 'Sors de là ! Dépêchons-nous !' partially split ('nous.'); Whisper turbo captures ('Sonte à moi ! Épéche-nous !'), v3 omits. |
| W01d | PARTIAL | Recap/action boundary split across c00; Whisper v3 clean. |
| W02a | PARTIAL | 'Il doit pen— il doit' pulled -38s early in Nemotron CTC; Whisper v3 confirms true acoustic interval 238.5-241.5s. |
| W02b | FAIL | Tao/Zia alternating child turns still collapsed into SPK_07 in Nemotron AND into SPEAKER_00 in pyannote community-1. |
| W02c | PASS | c02/c03 270s boundary duplicate cleanly reconciled across chunks (no cross-chunk echo emitted). |
| W02d | PARTIAL | Intra-chunk speaker mixing around 280-308s persists in Nemotron; c03 displaced compressed echo caught in orig16/prior_def replays. |
| W03a | PASS | c10/c11 990s boundary duplicate cleanly reconciled; extended recovery promotes 8-unit continuation. |
| W03b | PARTIAL | CTC compression caught (17 units quarantined by ctc_multiword_temporal_compression at 44.44 chars/s + 9 cross_chunk_ctc_disagreement -> untimed utt-0148..0150 withheld from SRT/ASS); true timing remains unrecovered without acoustic alignment. |
| W03c | PARTIAL | Cross-speaker short 4w repetition preserved ambiguously rather than deleted. |
| W04a | PARTIAL | Bidirectional split ('Tu n'as pas trouvé' at 1203s, 'Esteban et Tao' at 1253s) persists in default Nemotron. |
| W04b | PASS | 1253.16-1255.155s continuous turn preserved without truncated core. |
| W04c | PARTIAL | 1264.36-1266.575s shout localized in default; vetted overlay re-anchors exact [1264.3, 1266.3]. |
| W05a | PASS | 1352.66-1353.909s 'Il joue avec des cailloux!' recovered in default run (and pyannote separates speaker; Whisper v3 missed it). |
| C1 | PASS | Clean 29-token mid-core control preserved intact with 0 quarantines. |
| C2 | PASS | Genuine repeat 'Je vais y arriver! Je vais y arriver!' + concurrent speaker preserved. |
| S01 | PARTIAL | Misheard 'Parlez à larguer' remains; speaker separation SPK_04/SPK_05 clean. |
| S02 | PASS | Default run captures full negation at true 355.31-359.849s (utt-0068/0069); prior_default_fr_job replay proves straddled_overlap_source_unlocalized fallback retains all 12 words untimed. |
| S03 | PARTIAL | 'emmuré! en murée!' duplication artifact remains in c12/c13. |
| S04 | PARTIAL | 1304.78-1316.39s speaker fragments ('Maintenant,' / 'il n'y') remain; audio review confirms normal speech rate (~15-16 chars/s), not 82.35 cps. |
| S05 | PASS | 1371.69-1374.477s Pichu turn cleanly separated as SPK_25 from Narrator SPK_26; 1327/1329 Alerte shouts both preserved. |
