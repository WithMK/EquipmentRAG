# GGUF Embedding Backend

EquipmentRAG can use the original Sentence Transformers model directory or a
GGUF embedding model served by llama.cpp. The default remains
`sentence_transformers`; existing configurations continue to work unchanged.

## Start bge-m3-Q4_0.gguf

BGE-M3 uses CLS pooling and produces 1024-dimensional, L2-normalized dense
vectors. Start a second llama-server process on a port different from the chat
LLM server:

```powershell
llama-server.exe `
  -m D:\OfflineAssets\models\embedding\bge-m3-Q4_0.gguf `
  --embedding `
  --pooling cls `
  --embd-normalize 2 `
  -ngl 999 `
  -c 8192 `
  -b 2048 `
  -ub 512 `
  --host 127.0.0.1 `
  --port 8081
```

The existing Qwen chat server may continue on port `8080`.

On Android/Termux, grant shared-storage access once with
`termux-setup-storage`, then use the Termux-built `llama-server`. Start with
CPU inference; add GPU-offload options only if that build supports them:

```bash
./llama-server \
  -m /storage/emulated/0/Download/bge-m3-Q4_0.gguf \
  --embedding \
  --pooling cls \
  --embd-normalize 2 \
  -c 8192 \
  -b 512 \
  -ub 256 \
  --host 127.0.0.1 \
  --port 8081
```

## Configure EquipmentRAG

Copy `config/config.gguf.example.yaml` to the ignored local configuration and
adjust the paths:

```powershell
Copy-Item .\config\config.gguf.example.yaml .\config\config.local.yaml
```

The relevant section is:

```yaml
embedding:
  backend: llama_cpp
  model_path: D:/OfflineAssets/models/embedding/bge-m3-Q4_0.gguf
  base_url: http://127.0.0.1:8081/v1
  model: bge-m3-q4
  dimension: 1024
  request_timeout_seconds: 120
  batch_size: 16
  device: null
  normalize_embeddings: true
```

`model_path` is validated locally so the configured GGUF release is recorded
with the deployment. llama-server, not Python, loads the file.

## Validate and reindex

```powershell
python -m app.embedding.embedding_service `
  --config .\config\config.local.yaml `
  --task query `
  --text "GGUF embedding 확인"
```

The command must report `vector_length: 1024`. Do not reuse a ChromaDB created
with the HF model. Use a separate ChromaDB path or back up and rebuild the
index, because HF and Q4 GGUF vectors must not be mixed.

```powershell
python -m app.indexer --config .\config\config.local.yaml --dry-run
python -m app.indexer --config .\config\config.local.yaml
```

Compare representative Korean equipment and C# queries against the HF index
before making the Q4 index the operational default.
