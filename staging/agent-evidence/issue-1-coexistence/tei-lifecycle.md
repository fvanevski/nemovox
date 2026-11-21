# `tei-qwen3-embedding` Container Identity & Lifecycle Verification

- **Exact Container Name:** `/tei-qwen3-embedding`
- **Exact Container ID:** `64c86ba8a443dddafc9ae268d7381bdbfe538f25aa74c2214d1d4863bad77b84` (`id_prefix=64c86ba8a443`)
- **Image / Supervisor / Ports:** `ghcr.io/huggingface/text-embeddings-inference:120-1.9`, Compose project `tei` / service `qwen3-embedding`, `RestartPolicy=unless-stopped`, `127.0.0.1:8005->8080/tcp`, GPU device request `capabilities=[["gpu"]]`.
- **Authoritative Run (`task-259`, `22:56:14Z`–`22:58:10Z`):**
  - `TEI_ORIGINAL_STATE`: `running`, `healthy`, `http://127.0.0.1:8005/health` `200`.
  - Because **Lane A (`TEI ON`) passed complete coexistence** (`COEXIST_PASS`), stopping TEI was **`NOT_NEEDED`** in the authoritative run.
  - `TEI_FINAL_STATE`: `running`, `healthy`, `http://127.0.0.1:8005/health` `200`, same container ID `64c86ba8a443dddafc9ae268d7381bdbfe538f25aa74c2214d1d4863bad77b84`.
- **Earlier Diagnostic Run Observation (`logs/diagnostic_lane_b_tei_check_sanitized.txt`):**
  - In a prior harness run (before adding the `VLLM_API_KEY` header to `/v1/models` polling, and where a transient external GPU spike dropped free VRAM to `8.44 GiB` during initial vLLM startup), `tei-qwen3-embedding` (`64c86ba8a443`) was cleanly stopped (`free_mib` `29,075 -> 30,809`, **`+1,734 MiB` reclaimed**), remained stopped without auto-restart while Lane B ran (`4,411 MiB` min free during Qwen), and was cleanly restored via `docker start 64c86ba8a443...` (`Running=true`, `Health=healthy`, HTTP `200` in `20.36s`), with `local_embedding.service` untouched (`active`).
