# Lab Vector Embeddings & IR Metrics

Welcome to the comprehensive documentation for the lab dedicated to **Vector Search**, **Embedding**, and **Information Retrieval (IR)**.

## 1. Background & Context

In most companies, knowledge is scattered across a multitude of documents in various formats:
* **Technical notes & maintenance manuals** (PDF files)
* **Reports & specifications** (DOCX files)
* **Architecture guides & IT documentation** (Markdown / MD files)

### The Business Challenge
Employees waste a considerable amount of time searching for the exact information they need. Traditional search methods (keywords, `Ctrl+F`, BM25) fall short in the following cases:
1. **Inability to handle synonyms:** A search for *“pipe leak”* will not return a document that mentions *“water pressure loss”*.
2. **Ignores intent:** A question phrased as a natural sentence (*“How do I work remotely on Fridays?”*) fails when faced with structured regulatory text (*“Two-day remote work agreements”*).
3. **Lack of semantic context:** Words are processed in isolation without understanding the overall meaning of the sentence.

---

## 2. Problem Statement & Solution Approach

### The Technical Problem
> **How can we enable a computer system to learn and evaluate the semantic meaning of heterogeneous text in order to instantly extract the most relevant answer, regardless of how the query is phrased?**

### The Chosen Approach: Dense Retrieval
Instead of matching keywords, we project all texts into a **multidimensional vector space**:
1. **Vectorization (Embedding):** A deep learning neural network (`all-MiniLM-L6-v2`) transforms each text segment into a dense 384-dimensional numerical vector.
2. **Geometric Proximity:** Texts that share a similar meaning are positioned close to one another in this space.
3. **Cosine Similarity Search:** The user’s query is itself converted into a 384-D vector. The system calculates the angle (cosine) between the query vector and all vectors in the database to extract the closest matches.

## 3. Business Value & ROI of the Solution

Beyond purely algorithmic aspects, semantic search based on embeddings delivers direct economic value:

### A. Operational Time Savings & Productivity (Direct ROI)
* **60% reduction in search time:** Employees spend an average of 1.8 hours per day searching for information. Vector search allows them to find the exact passage in just a few seconds.
* **Faster incident resolution:** Maintenance technicians or IT support staff can find the exact procedure without having to reread the entire technical documentation.

### B. Reducing Errors and Helpdesk Costs
* **Reducing the Workload on Level 2/Level 3 Support:** By automatically answering common questions (HR procedures, validation criteria, security), the workload on expert teams is significantly reduced.
* **Reduction in Costly Outages:** Immediate access to the correct operational instructions in the field minimizes mistakes and unplanned downtime.

### C. Knowledge Capitalization & Transferability (Knowledge Management)
* **Leveraging Unstructured Data:** Immediate use of PDF, Word, and Markdown formats without the need for re-entry or prior manual structuring.
* **Onboarding New Employees:** Accelerating the learning curve for new hires through a one-stop shop for accessing company knowledge.

---

## 4. Contributions & Added Value of the Lab

This lab provides the key framework for building a RAG (*Retrieval-Augmented Generation*) system:
* **Semantic Accuracy:** Accounts for synonyms and natural language queries.
* **Extensible Architecture:** Clear separation between the processing pipeline (FastAPI) and the user interface (Angular).
* **Integrated Metrics Evaluation:** Objective measurement of quality using scientific IR KPIs.

---

## 5. Technologies Used in the Lab

---
## 6. Detailed Technical Concepts: Chunks & Overlap

### A. Chunking
An embedding model has a **context window** (e.g., a maximum of 512 tokens). Attempting to vectorize an entire 20-page document presents two major drawbacks:
1. **Exceeding the model’s memory limit.**
2. **Semantic dilution:** The vector becomes an overly generalized average in which specific details are lost.

**Chunking** involves segmenting the document into fixed-size blocks (e.g., 150 words) to preserve the precision of the information.

### B. Concepts of Overlap (Overlap)
If the segmentation is strict, a key sentence located exactly at the boundary between two chunks may be split in two. **Overlap** (a 30-word overlap) involves duplicating the end of Chunk N at the beginning of Chunk N+1, ensuring that no critical information is truncated at the boundaries.

---

## 7. Evaluation Metrics (KPIs)

1. **Precision@K:** Accuracy ratio of the first K results (e.g., K=3).
2. **MRR (Mean Reciprocal Rank):** Evaluates the system’s ability to place the best result at the very top of the list.
3. **Cosine Similarity Score:** Defines the proximity in orientation between the query vector and the document vector (recommended threshold ≥ 0.45).

---

## 8. Execution Guide

### Launching the Backend (Python)
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000