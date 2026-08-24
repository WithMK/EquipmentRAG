# Office Document RAG implementation plan

## Scope

Add native `.pptx` and `.xlsx` extraction to the existing Document RAG without
changing Code RAG defaults or the `document_chunks` collection contract.

## Design

1. Normalize slides, sheets, tables, notes, values, and formulas into
   `DocumentBlock` instances.
2. Extend chunk traceability with an optional Excel `cell_range` while retaining
   backward compatibility with existing Chroma metadata.
3. Register both parsers in the shared parser registry and configuration examples.
4. Reuse incremental indexing so file or sidecar changes trigger only the affected
   document.
5. Surface slide, sheet, and cell range in search and grounded-answer sources.
6. Validate with real generated PPTX/XLSX containers and the full regression suite.

## Deferred

OCR, picture analysis, chart semantics, Excel calculation, legacy `.ppt`/`.xls`,
hybrid retrieval, reranking, and context orchestration remain separate phases.
