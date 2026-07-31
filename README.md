## 2026-07-22

### Goal

Implement core data models.

### Progress

Finished Document, Chunk and RetrievedChunk classes.

### Decisions

Metadata stored as a dictionary to support future extensions.

### Problems

None.

Implement document loaders.

### 2026-07-25

## Session: Recursive Chunker

### Goal

Implement the baseline chunking strategy.

### Design Decisions

- RecursiveCharacterTextSplitter selected as the baseline.
- Chunk size and overlap moved to config.py.
- Every chunk stores metadata for traceability.

### Reflection

Recursive chunking preserves semantic boundaries better than fixed-size chunking and provides a strong baseline for later adaptive strategies.

### 2026-07-30

                ┌────────────────────┐
                │ Document Loader    │ ✅
                └─────────┬──────────┘
                          │
                          ▼
                ┌────────────────────┐
                │ Recursive Chunker  │ ✅
                └─────────┬──────────┘
                          │
                          ▼
                ┌────────────────────┐
                │ Embedding Manager  │ ✅
                └─────────┬──────────┘
                          │
                          ▼
                ┌────────────────────┐
                │ ChromaDB Store     │ ✅
                └─────────┬──────────┘
                          │
                          ▼
                     Retriever

### 2026-07-31

                   User Query
                        │
                        ▼
                Query Analyzer
                        │
                        ▼
               QueryAnalysis
                        │
                        ▼
                Decision Engine
                        │
        ┌───────────────┼────────────────┐
        ▼               ▼                ▼

Retrieval Type Top-K Chunk Strategy
▼ ▼ ▼
Retriever Pipeline

