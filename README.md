# Chat with PDF: a simple RAG pipeline

Ask questions about any PDF and get answers grounded in the document, with page citations.
Tested on the *Cognizant Forward Team 2027* student brochure.

## How it works

```
PDF ──► pages ──► chunks (120 words, 30 overlap) ──► FastEmbed vectors ──► NumPy index
                                                                              │
question ──► FastEmbed vector ──► cosine similarity ──► top-4 chunks ─────────┘
                                                              │
                                     prompt ("answer only from context, cite pages")
                                                              │
                                                    Groq LLM ──► answer + [p. N] citations
```

| Step | Choice | Why |
|---|---|---|
| Loading | pypdf, page by page | keeps page numbers for citations |
| Chunking | 120 words, 30-word overlap | specific chunks; overlap keeps sentences whole |
| Embeddings | FastEmbed `BAAI/bge-small-en-v1.5` | free, local, fast on CPU |
| Vector index | in-memory NumPy matrix, cosine similarity | small document, so no vector database needed |
| LLM | Groq `openai/gpt-oss-120b`, temperature 0 | free tier, fast, deterministic |
| Guardrail | answer only from context, fixed refusal message | reduces hallucination |
| Evaluation | Hit@k and MRR on 8 labelled questions | checks retrieval finds the right page |

## Run it (VS Code)

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (Mac/Linux: source .venv/bin/activate)
pip install -r requirements.txt
copy .env.example .env          # then paste your free Groq key into .env
```

Open `notebooks/chat_with_pdf.ipynb`, select the `.venv` kernel and click **Run All**.
Without a Groq key the notebook still runs and shows the retrieved passages instead of an LLM answer.

To use your own document, put a PDF in `data/` (the first PDF found is used).

## Possible improvements

Hybrid search (BM25 + vectors), a cross-encoder reranker, a vector database (Qdrant / FAISS) for many documents, and a Streamlit UI.
