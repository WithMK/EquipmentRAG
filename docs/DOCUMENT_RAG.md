# Document RAG Phase 1

EquipmentRAG can index and retrieve DOCX, text-based PDF, PPTX, XLSX, Markdown, and TXT
documents without changing the existing C# Code RAG path. Documents use the
`document_chunks` ChromaDB collection while code continues to use the collection
configured under `chromadb.collection_name`.

## Configuration

Enable the tracked example only in a local ignored configuration file.

```yaml
document:
  enabled: true
  source_paths:
    - D:/EquipmentData/documents
  extensions: [.docx, .pdf, .pptx, .xlsx, .md, .txt]
  exclude_directories: [.git, .vs, archive]
  chunk_size: 3000
  chunk_overlap: 300
  collection_name: document_chunks
```

Actual documents must remain outside the repository. OCR and image-only PDF
files are not supported in this phase.

## Revision sidecar

Optional metadata is supplied next to a document as either
`Document.docx.metadata.yaml` or `Document.metadata.yaml`. The file-specific
name is checked first.

```yaml
document_id: trimming-loader-spec
project: TrimProject
equipment: Trimming
unit: Loader
document_type: Specification
title: Trimming Loader Specification
revision: Rev.3
document_status: active
is_latest: true
created_date: 2026-08-01
modified_date: 2026-08-20
```

Use `document_status: obsolete` and `is_latest: false` for superseded revisions.
A sidecar change participates in the incremental state hash and triggers only
that document's reindexing.

## Index

Review a dry run before persistent indexing.

```powershell
python -m app.document_indexer `
  --config config\config.local.yaml `
  --dry-run

python -m app.document_indexer `
  --config config\config.local.yaml
```

Use `--full` after an intentional parser, chunking, metadata-schema, or embedding
model change. `--source` may be repeated to override configured roots locally.

## Retrieve

The default query includes the configured equipment plus `active` and latest
metadata filters.

```powershell
python -m app.document_search "Loader Vacuum 관련 사양" `
  --config config\config.local.yaml `
  --unit Loader `
  --document-type Specification
```

An explicit revision automatically removes the latest-only restriction. Use
`--include-obsolete` for unrestricted historical retrieval.

```powershell
python -m app.document_search "변경 전 Loader 사양" `
  --config config\config.local.yaml `
  --revision Rev.1 `
  --json
```

Python integrations should call the retrieval layer directly; no LLM is needed.

```python
from app.config import load_config
from app.retrieval.document_retriever import DocumentRetriever

retriever = DocumentRetriever(load_config("config/config.local.yaml"))
results = retriever.retrieve(
    "Loader 제품 감지 Interlock 조건",
    top_k=5,
    equipment="Trimming",
    unit="Loader",
    document_type="Specification",
)
```

Each result contains the chunk ID, score, text, source path, and complete
metadata including revision, section, page, slide, sheet, and cell-range fields.

## Optional grounded answer

`RagService` remains code-first by default. Select document mode explicitly for
local answer-generation tests with the existing llama.cpp or Ollama provider.

```powershell
python -m app.rag_service "Loader Interlock 조건을 설명해줘." `
  --source-type document `
  --config config\config.local.yaml `
  --unit Loader `
  --document-type Specification `
  --include-source-text
```

The retrieval interface is the production integration point for a future Context
Orchestrator. OCR, chart interpretation, hybrid search, and reranking can be added
without changing the normalized parser-to-chunker contract.
