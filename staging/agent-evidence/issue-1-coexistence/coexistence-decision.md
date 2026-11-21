# Coexistence Decision & Operational Recommendation

## Overall Verdict: **`COEXIST_PASS_TEI_ON`**

On the single RTX 5090 (`32,607 MiB`, UUID `GPU-c75b2559-b8e1-ebf9-fe82-65cd2b3ca4c0`), **Nemotron 3.5 Transcribe 30B-A4B NVFP4 vLLM** (`gpu_memory_utilization=0.78`, `24,406 MiB` resident) and **Qwen3-ForcedAligner-0.6B CUDA BF16** (`2,564 MiB` resident) **successfully coexist simultaneously in `--qwen-gpu-mode available` even with `tei-qwen3-embedding` (`1,758 MiB`) and desktop `kwin_wayland` (`327 MiB`) running**.

- **Observed Margin:** Peak simultaneous usage reached `30,061 MiB` / `32,607 MiB`, leaving **`2,066 MiB`** free VRAM during Qwen alignment (`4,637 MiB` free prior to Qwen launch, above `_qwen_align_worker.py`'s `2,600,000,000`-byte admission threshold).
- **Operational Recommendation (`--qwen-gpu-mode available`):**
  - Use `--alignment-fallback qwen --qwen-gpu-mode available --qwen-device cuda:0` directly alongside the running Nemotron ASR vLLM server when baseline free VRAM before Qwen launch is >= `2.6 GB` (no managed ASR takeover or systemd ownership infrastructure is required for this single-GPU profile when no other heavy GPU models are loaded).
  - If another GPU consumer (such as a transient model or SwiftU) reduces free VRAM below `2.6 GB` while ASR is resident, operators may temporarily stop `tei-qwen3-embedding` (`+1,734 MiB` reclaimed, verified safe/reversible) before retrying `--qwen-gpu-mode available`.
- **Known Limitations:**
  - Measured on a bounded `18.56s` (2-chunk) live ASR + 2-span Qwen workload; indefinite concurrent heavy multi-request load or simultaneous heavy GPU SwiftU/LLM workloads were not tested.
  - Note that in an earlier diagnostic run (`22:51:40Z`), an unrelated transient external GPU allocation left only `8.44 GiB` free during vLLM startup (`vLLM` failed cleanly on `request_memory`); coexistence depends on other uncontrolled GPU processes not transiently consuming the remaining headroom.
  - `--qwen-gpu-mode managed` remains `NOT_TESTED` live (no `nemovox-asr.service` or cooperative lease exists on host).
- **Issue #1 Recommendation (`CLOSE_WITH_DOCUMENTED_OPTIONAL_MODE_LIMITATION`):**
  - Core issue #1 requirements (selective subtitle-only fallback, reusable CUDA BF16 Qwen worker, safe default `off`, intact CTC/source/speakers, verified RTX 5090 coexistence in `available` mode, and CPU/mocked managed-handoff safety gates) are demonstrated; live `--qwen-gpu-mode managed` can remain explicitly documented as unexercised without an owned ASR systemd unit/lease unless Central decides to pursue separate launcher/unit integration.
