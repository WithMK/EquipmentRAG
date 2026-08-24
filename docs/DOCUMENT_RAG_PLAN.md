# Document RAG Phase 1 implementation plan

## Constraints

- Preserve the existing C# indexer, search CLI, RAG service defaults, and tests.
- Keep code and document parsing, chunking, state, and collections separate.
- Reuse the local BGE-M3 embedding service, persistent ChromaDB wrapper, and LLM clients.
- Keep runtime operation offline and do not commit real documents, source, models, indexes, or credentials.

## Architecture

```text
DOCX / text PDF / Markdown / TXT
  -> document scanner and format parser
  -> NormalizedDocument
  -> heading-aware DocumentChunker
  -> BGE-M3
  -> document_chunks collection
  -> DocumentRetriever
  -> optional RagService answer generation
```

The existing code path continues to use `ChunkMetadata` and the configured code
collection. The ChromaDB wrapper accepts a metadata codec type so the new
`DocumentChunkMetadata` can use the same persistence and query implementation.

## Delivery steps

1. Add typed document source, normalized block, chunk, and metadata models.
2. Add optional document configuration without changing existing `AppConfig` callers.
3. Generalize Chroma metadata serialization while retaining code defaults.
4. Add DOCX, text PDF, Markdown, and TXT parsers plus a parser registry.
5. Add structural document chunking with heading paths and bounded fallback splitting.
6. Add a lazy, incremental document indexer and separate state manifest.
7. Add a library-first document retrieval interface and document search CLI.
8. Extend `RagService` with an opt-in document source type.
9. Add regression, parser, chunker, indexer, retrieval, and RAG tests.
10. Update configuration, usage, dependencies, and offline wheel guidance.

## Deferred work

XLSX, PPTX, OCR, vision, hybrid keyword search, reranking, Roslyn, orchestration,
and document generation remain outside this phase. The normalized block schema
retains table, page, slide, and sheet fields so those formats can be added later.
