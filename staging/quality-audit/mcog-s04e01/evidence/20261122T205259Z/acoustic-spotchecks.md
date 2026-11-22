# Acoustic Spot-Check Verification Record

## Methodology & Boundaries
- **Extraction:** Read-only `ffmpeg` seek/decode (`-vn -ac 1 -ar 16000`, 16 kHz mono PCM WAV) of `18` bounded `8-28s` subclips from `<INPUT_VIDEO>` into private `chmod 700` scratch storage. Zero GPU inference, zero model startup, zero full-episode reprocessing, and zero audio bytes published.
- **Perception Method:** Direct multimodal audio inspection via Antigravity `read_file` on each short WAV clip, cross-referenced against subclip start offsets and raw CTC token timestamps.
- **Epistemic Hygiene:** Speech presence, approximate true anchor, speaker voice identity, and wording discrepancies are separated; uninspected regions of the 23-minute episode are explicitly left unverified.

## Spot-Check Outcomes

| Anchor Clip (s) | Case(s) | Spoken Anchor (s) | Outcome | Speech / Speaker / Timing Observation |
|---|---|---|---|---|
| `25-45`, `40-55`, `45-65`, `65-86`, `78-88` | `W01a, W01b, W01c, W01d` | `~27-37`, `~43.5-59`, `~57.5-59.5`, `~59.5-62`, `~77.5-78.2` | **HEARD / NOT_HEARD** | Theme song vocals repeat twice in `27-37s`; distorted name is absent at `82.7s`. Narrator recap alone at `43.5-59s`; `"Les chèvres"` is **NOT_HEARD** at `43-49s` and **HEARD** right at `~77.5-78.2s`. Shout over dramatic music **HEARD** at `~57.5-59.5s` (partially masked wording near `"Dépêchons-nous!"`, omitted by ASR). `"Nos héros se lancent à sa poursuite"` **HEARD** once at `~59.5-62s` and **NOT_HEARD** at `80-82.5s`. |
| `198-220` | `W02a, W02b` | `~196-220` | **HEARD / NOT_HEARD** | Continuous Tao monologue -> Tao/Zia rapid alternating turn pair (`~214-220s`, clearly distinct male & female voices both collapsed into `SPK_07`). False-start tokens (`202-210s`) are **NOT_HEARD** here. |
| `240-264`, `262-290`, `290-312` | `W02a, W02c, W02d` | `~240-242.5`, `~243-263`, `~264.5-279`, `~280-312` | **HEARD / NOT_HEARD** | `"Il doit penser à Isabella Laguerra"` **HEARD** at `~240-242.5s`. Pedro & Sancho occupy `243-263s`; Zia's `W02c` monologue is **NOT_HEARD** in `243-262s`, **HEARD ONCE** in `264.5-279s` (crossing the `270s` boundary cleanly), and **NOT_HEARD** in `282-307s`, where Mendoza, Zia (`~292-296s`), Sancho, and Pedro speak different alternating lines collapsed across `SPK_08/10/07`. |
| `982-1004` | `W03a` | `~985.5-999` | **HEARD** | Priest speech (`~985.5-996s`) spoken once; Zia shout overlaps near `996-999s`. Supports `c11`'s `+9s` displacement at `996.83s`. |
| `1004-1016`, `1012-1022`, `1015-1027`, `1020-1036` | `W03b, C2` | `~1004-1014`, `~1022.5-1026`, `~1028-1034` | **HEARD / NOT_HEARD** | Priest's question + Mendoza's reply + Zia's search line occur at `~1004-1014s`, **NOT_HEARD** in `1017-1029s` (silently displaced `+12-16s` and merged under `SPK_04`). Genuine repeated exclamation (`C2`) **HEARD** clearly by young male voice at `~1022.5-1026s`. |
| `1070-1090` | `W03c` | `~1073-1075` | **HEARD / NOT_HEARD** | `"Oh non ! Trop tard !"` **HEARD ONCE** at `~1073-1075s` and **NOT_HEARD** at `1085.5-1087.1s` (`c12` duplicate disproven). |
| `1200-1216`, `1228-1246`, `1244-1268` | `W04a, W04b, W04c` | `~1230-1232.8`, `~1253-1255.3`, `~1264.4` | **HEARD / NOT_HEARD** | Only grunts/SFX in `1200-1216s` (`Tu n'as pas trouvé...` **NOT_HEARD** at `1203s`, **HEARD** at `~1230-1232.8s` right before `1233s` reply; proves `-27s / +17s` split rather than catalog `+45s`). Complete `W04b` exclamation **HEARD** at `~1253-1255.3s`. `W04c` (`Mes amis...`) **HEARD** at `~1264.4s` (**NOT_HEARD** at `1254.6s`). |
| `1344-1364` | `W05a` | `~1352.0-1354.0` | **HEARD** | Question/narrator through `~1352s`, then distinct child voice clearly **HEARD** saying the `6`-token pebble line at `~1352.0-1354.0s`, immediately answered by narrator at `~1354.2s`. Confirms `c14` right context accurate and `chunking.py:377` `1.5s` bound dropped live speech. |
| `686-700` | `C1` | `~687.5-696.5` | **HEARD** | Single speaker continuous `29`-token line matching `utt-0112` boundaries cleanly with following pause + shout at `~698s`. |
| `1300-1315`, `1326-1334`, `1360-1375` | `S04`, alt-C2, `S05` | `~1304.5-1315`, `~1327-1330`, `~1367-1374` | **HEARD** | `S04` spoken continuously without gap; `1327-1330s` rapid shouted double `"Alerte!"`; `S05` clearly switches from adult male narrator (`~1367.5-1369.5s`) to high-pitched parrot Pichu (`~1369.5-1374s`) before cutoff at `1370.63s`. |

## Acoustic Verification Gaps
- Non-priority spans (e.g. `01:32-02:39`, `05:26-08:24`, `13:24-13:51`, `18:10-18:27`) were not extracted/listened to (`NOT_CHECKED` acoustically).
- Burned-in on-screen text (`D7` `00:35`) is out of audio-only scope.
- Independent full-episode speaker diarization annotation (`"~8 speakers"`) does not exist (`UNVERIFIED`).
