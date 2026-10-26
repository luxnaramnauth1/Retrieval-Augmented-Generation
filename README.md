Retrieval-augmented generation (RAG) is a pattern where a language model answers questions using documents that are fetched at query time, rather than relying only on what it learned during training.

A typical RAG pipeline has three steps. First, documents are split into chunks. Second, each chunk is turned into an embedding and stored in an index. Third, when a user asks a question, the system finds the most similar chunks and includes them in the prompt sent to the model.

RAG is useful when the information is private, recent, or too large to fit in a single prompt. It also makes answers easier to check, because the system can point to the source passages it used.

Common failure modes

Poor chunking is a frequent problem. If chunks are too small, they lose context. If they are too large, the retrieved text contains a lot of irrelevant material. Retrieval quality also depends on the embedding model, so testing a few models on your own data is worthwhile.
