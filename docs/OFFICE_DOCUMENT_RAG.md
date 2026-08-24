# Office Document RAG

EquipmentRAG extends the existing `document_chunks` pipeline with native PPTX
and XLSX parsing. The Office formats reuse the same scanner, sidecar metadata,
incremental state, embedding service, retrieval filters, and grounded-answer mode.

## Supported extraction

PPTX extraction preserves:

- slide title and body text;
- text in grouped shapes;
- tables in row order;
- speaker notes;
- one-based slide numbers.

XLSX extraction preserves:

- workbook title and worksheet names;
- contiguous non-empty row regions as tables;
- cell ranges such as `A1:D12`;
- scalar values, dates, and booleans;
- formula text and a cached value when one exists in the file.

Office files use the same optional YAML sidecars as other documents. Add the
extensions to the document configuration:

```yaml
document:
  enabled: true
  extensions: [.docx, .pdf, .pptx, .xlsx, .md, .txt]
```

Run the existing dry run before indexing:

```powershell
python -m app.document_indexer `
  --config config\config.local.yaml `
  --dry-run

python -m app.document_indexer `
  --config config\config.local.yaml
```

Search results and RAG sources expose `slide`, `sheet`, and `cell_range`. The
original source path and revision metadata remain available for every chunk.

## Intentional limits

- `.ppt` and `.xls` legacy binary formats are not supported.
- Pictures and scanned text are not OCR processed.
- Chart series are not interpreted semantically.
- Excel formulas are not recalculated. Stored cached values are used when present.
- Password-protected or corrupt Office files fail with a parser error.
- Sheets larger than the parser safety limit must be split before indexing.

Keep Office files under an ignored local or internal document root. Never commit
real equipment documents, generated ChromaDB data, or model files to GitHub.
