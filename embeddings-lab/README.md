# Embeddings Lab

This is a small personal lab for learning how vector embeddings and semantic search work in practice. The goal is not to ship a product, it is to understand what happens between "a user types a question" and "the system returns relevant passages".

## Why this exists

Keyword search has real limits. A search for "pipe leak" will not match a document that talks about "loss of water pressure", even though a human reading both would immediately see the connection. Keyword search also tends to return whole documents, leaving the reader to hunt through pages to find the one relevant paragraph.

Embeddings address both problems. Instead of matching words, the text is converted into a vector, a list of numbers that represents its meaning. Two pieces of text that mean similar things end up close to each other in that vector space, even if they do not share a single word. Searching then becomes a matter of finding which vectors are closest to the vector of the query.

## What the lab actually does

1. A set of documents (PDF, DOCX, Markdown) is loaded from `backend/datasets/`.
2. Each document is split into chunks, since an embedding model can only look at a limited amount of text at once and a single vector for an 80 page report would be too vague to be useful.
3. Each chunk is turned into a 384 dimension vector using `all-MiniLM-L6-v2` (a small, fast sentence transformer model).
4. Vectors are kept in memory. There is no database here, this is intentionally the simplest possible setup.
5. On a search request, the query is embedded the same way, then compared to every stored chunk using cosine similarity. The closest chunks are returned, ranked by score.

This is the retrieval half of what is usually called RAG (Retrieval-Augmented Generation). The lab stops there on purpose: it returns raw passages, not a written answer. Feeding those passages to a language model so it can write a proper answer in natural language is the generation half, and it is a natural next step for this project, but it is not part of what is being learned or tested here. Keeping retrieval isolated makes it much easier to see whether the search itself is actually good, without a language model smoothing over or hiding a bad result.

## Two ideas worth understanding: chunking and overlap

Chunking is how a document gets split before embedding. Too large and the resulting vector becomes an average of too many topics, so it stops matching anything precisely. Too small and you lose context. A common starting point is a few hundred words per chunk.

Overlap means the end of one chunk is repeated at the start of the next one. Without it, a sentence that happens to sit right on a chunk boundary gets cut in half and its meaning can be lost in both pieces. A small overlap (a few dozen words) fixes that at the cost of some duplicated content in the index.

## How search quality gets measured

Once retrieval is in place, the natural question is: is it any good? Three metrics come up often:

Precision@K looks at the top K results and asks how many of them are actually relevant.

MRR (Mean Reciprocal Rank) rewards systems that put the correct answer near the top of the list, not just somewhere in it.

The cosine similarity score itself gives a rough sense of confidence. In this lab, scores above roughly 0.45 tend to indicate a real match, though this threshold depends heavily on the model and the data.

## Project layout

```
embeddings-lab/
  README.md
  backend/
    main.py            FastAPI app: extraction, chunking, embedding, search endpoint
    datasets/           source documents used to build the in memory index
    requirements.txt
    Makefile
    Dockerfile
```

## Running it locally

You need Python 3.13 or newer.

```bash
cd backend
make install
make run
```

`make run` starts the API on port 5050 with auto reload. `make dev` does the same thing directly through uvicorn if you prefer. Either way, once it is up:

```
http://localhost:5050/docs
```

opens the interactive Swagger UI, where `POST /search` can be tried directly.

## Running it with Docker

```bash
cd backend
make docker-build
make docker-run
```

The embedding model is downloaded once at build time and baked into the image, so starting the container does not require a network call to Hugging Face. The API is reachable on `http://localhost:5050` once the container is running.

## Calling the search endpoint

```bash
curl -X POST http://localhost:5050/search \
  -H "Content-Type: application/json" \
  -d '{"query": "how does remote work policy handle Fridays", "top_k": 3}'
```

The response is a ranked list of the most relevant chunks, each with its source filename, the text itself, and its similarity score. No answer is generated, the point of this lab is to see the retrieval step working on its own.

## What is deliberately out of scope

There is no vector database, no reranking step, no authentication, and no generation step. All of that is normal for a production RAG system, but adding it here would make it harder to isolate what embeddings and retrieval alone can and cannot do, which is the actual point of this lab.
