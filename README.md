# Medical-AI Bot

Medical-AI Bot is a secure, role-aware healthcare assistant that combines document RAG, SQL analytics, metadata-based access control, hybrid retrieval, reranking, and cloud LLM answer generation.

The system is designed for a healthcare environment where different staff roles must only access the document collections and analytics capabilities they are authorized to use.

![Medical-AI Bot Architecture](docs/medical_ai_bot_architecture_flowchart.png)

---

## Key Features

- Role-aware login using demo healthcare users.
- Signed access token authentication.
- FastAPI backend with required API endpoints.
- Next.js frontend with login, role badge, accessible collections, chat UI, retrieval type display, and source citations.
- Document ingestion using Docling.
- Hierarchical chunking for PDF and Markdown files.
- Chunk metadata for source document, collection, access roles, section title, and chunk type.
- Dense embedding using `BAAI/bge-small-en-v1.5`.
- Sparse retrieval using FastEmbed BM25.
- Local Qdrant vector store.
- Hybrid dense + sparse document retrieval.
- RBAC filtering at the Qdrant retrieval layer using metadata filters.
- Cross-encoder reranking.
- SQL RAG over `mediassist.db`.
- Read-only SQLite query execution.
- Gemini cloud LLM integration using `gemini-3.5-flash-lite`.
- Source citations returned with document answers.
- Clear blocked-access responses for unauthorized requests.

---

## Tech Stack

| Layer            | Technology                            |
| ---------------- | ------------------------------------- |
| Backend API      | FastAPI                               |
| Frontend         | Next.js, TypeScript                   |
| Document Parsing | Docling                               |
| Vector Database  | Qdrant local mode                     |
| Dense Embeddings | FastEmbed, `BAAI/bge-small-en-v1.5`   |
| Sparse Retrieval | FastEmbed BM25                        |
| Reranking        | FastEmbed cross-encoder reranker      |
| SQL Database     | SQLite                                |
| Cloud LLM        | Gemini API, `gemini-3.5-flash-lite`   |
| Language         | Python 3.12                           |
| Package Checks   | `pip check`, `pip-audit`, `npm audit` |

---

## Role Access Matrix

| Role                | Accessible Document Collections                          | SQL Analytics |
| ------------------- | -------------------------------------------------------- | ------------- |
| `doctor`            | `general`, `clinical`, `nursing`                         | No            |
| `nurse`             | `general`, `nursing`                                     | No            |
| `billing_executive` | `general`, `billing`                                     | Yes           |
| `technician`        | `general`, `equipment`                                   | No            |
| `admin`             | `general`, `clinical`, `nursing`, `billing`, `equipment` | Yes           |

Access control is enforced at the retrieval layer using Qdrant metadata filters. The frontend does not decide access permissions.

---

## Demo Users

| Username       | Password            | Role                |
| -------------- | ------------------- | ------------------- |
| `dr.mehta`     | `doctor`            | `doctor`            |
| `nurse.priya`  | `nurse`             | `nurse`             |
| `billing.ravi` | `billing_executive` | `billing_executive` |
| `tech.anand`   | `technician`        | `technician`        |
| `admin.sys`    | `admin`             | `admin`             |

These accounts are for demonstration only.

## Environment Setup

Create and activate a virtual environment:

```powershell
cd C:\medical-ai-bot
python -m venv .venv
.venv\Scripts\activate
```

Install backend dependencies:

```powershell
pip install -r backend\requirements.txt
```

Install frontend dependencies:

```powershell
cd C:\medical-ai-bot\frontend
npm install
```

---

## Environment Variables

Create a local `.env` file in the project root.

Use `.env.example` as the reference.

Required Gemini configuration:

```env
LLM_PROVIDER="gemini"
LLM_MODE="cloud"
GEMINI_API_KEY="your-gemini-api-key"
GEMINI_MODEL="gemini-3.5-flash-lite"
GEMINI_TEMPERATURE=0.2
GEMINI_MAX_OUTPUT_TOKENS=2048
```

Qdrant local configuration:

```env
QDRANT_MODE="local"
QDRANT_LOCAL_PATH="data/vector_store/qdrant"
QDRANT_COLLECTION_NAME="medical_ai_bot_chunks"
DENSE_VECTOR_NAME="dense"
SPARSE_VECTOR_NAME="bm25_sparse"
DENSE_VECTOR_SIZE=384
```

Frontend environment file:

```text
frontend/.env.local
```

```env
NEXT_PUBLIC_API_BASE_URL=http://127.0.0.1:8000
```

Do not commit `.env` or `.env.local`.

---

## Backend API Endpoints

| Method | Endpoint              | Purpose                                        |
| ------ | --------------------- | ---------------------------------------------- |
| `GET`  | `/health`             | Health check                                   |
| `POST` | `/login`              | Demo user login                                |
| `GET`  | `/collections/{role}` | Returns collections available for a role       |
| `POST` | `/chat`               | Main chat endpoint for document RAG or SQL RAG |

---

## Chat Response Format

The `/chat` endpoint returns:

```json
{
  "answer": "There are 12 rejected claims.",
  "sources": [],
  "retrieval_type": "sql_rag",
  "role": "billing_executive"
}
```

Document RAG responses include source citations:

```json
{
  "answer": "Based on the retrieved documents...",
  "sources": [
    {
      "source_document": "drug_formulary.pdf",
      "section_title": "Approved Drug Formulary",
      "collection": "clinical"
    }
  ],
  "retrieval_type": "hybrid_rag",
  "role": "doctor"
}
```

Blocked responses return a clear access message:

```json
{
  "answer": "As a nurse, you do not have access to SQL analytics. I can answer document questions from these collections: general, nursing.",
  "sources": [],
  "retrieval_type": "blocked",
  "role": "nurse"
}
```

---

## How the Query Flow Works

### 1. Login

The user logs in through the Next.js frontend using one of the demo accounts.

The FastAPI backend validates the credentials and returns a signed access token containing the username and role.

### 2. Chat Request

The frontend sends the question and access token to:

```text
POST /chat
```

The backend extracts the role from the signed token.

### 3. Query Routing

The backend routes the question to one of two paths:

| Query Type          | Route               |
| ------------------- | ------------------- |
| Analytical question | SQL RAG             |
| Document question   | Hybrid document RAG |

Examples of analytical questions:

```text
How many claims are rejected?
Show claim status distribution.
Which insurer has the highest approved amount?
```

Examples of document questions:

```text
What drug formulary guidance is available for antibiotic use?
What infection control guidance should nurses follow?
What equipment maintenance guidance is available?
```

### 4. SQL RAG Path

For allowed roles, SQL RAG works as follows:

1. Gemini generates a safe SQLite `SELECT` query.
2. The query is validated as read-only.
3. SQLite executes the query against `mediassist.db`.
4. Gemini converts the result into a natural-language answer.

Only these roles can use SQL analytics:

```text
billing_executive
admin
```

### 5. Document RAG Path

Document RAG works as follows:

1. Docling parses source documents.
2. Documents are split into hierarchical chunks.
3. Chunks are stored with metadata.
4. Dense and sparse embeddings are generated.
5. Qdrant performs hybrid dense + sparse retrieval.
6. RBAC filters restrict retrieval by `access_roles` and `collection`.
7. Cross-encoder reranker selects the most relevant chunks.
8. Context builder prepares LLM-ready context with citations.
9. Gemini generates a grounded answer from the retrieved context.
10. Sources are returned with the answer.

---

## Chunk Metadata

Each document chunk includes:

```json
{
  "source_document": "drug_formulary.pdf",
  "collection": "clinical",
  "access_roles": ["doctor", "admin"],
  "section_title": "Approved Drug Formulary",
  "chunk_type": "text"
}
```

Supported chunk types:

```text
text
table
heading
code
```

---

## Build and Index Documents

Run these commands from the project root:

```powershell
python -m scripts.inspect_dataset
python -m scripts.verify_core_config
python -m scripts.build_document_inventory
python -m scripts.build_document_chunks
python -m scripts.setup_qdrant_collection --recreate
python -m scripts.index_document_chunks --recreate
```

---

## Validation Commands

Run these commands from the project root.

### Gemini connection

```powershell
python -m scripts.test_gemini_llm
```

Expected confirmation:

```text
Gemini connection successful.
Model: gemini-3.5-flash-lite
Provider: gemini
```

### SQL RAG allowed role

```powershell
python -m scripts.test_sql_rag --role billing_executive --question "How many claims are rejected?"
```

Expected answer:

```text
There are 12 rejected claims.
```

### SQL RAG blocked role

```powershell
python -m scripts.test_sql_rag --role nurse --question "How many claims are rejected?"
```

Expected access result:

```text
Access allowed: False
Role 'nurse' is not allowed to use SQL analytics.
```

### RBAC hybrid retrieval

```powershell
python -m scripts.test_rbac_hybrid_retrieval
```

Expected confirmation:

```text
ALL RBAC HYBRID RETRIEVAL TESTS PASSED
```

### Reranking and context

```powershell
python -m scripts.test_reranking_context
```

Expected confirmation:

```text
RERANKING CONTEXT CHECK: PASSED
```

### Backend API

```powershell
python -m scripts.test_backend_api
```

Expected confirmation:

```text
BACKEND API CHECK: PASSED
```

---

## Run the Backend

From the project root:

```powershell
.venv\Scripts\activate
python -m uvicorn backend.app.main:app --reload
```

Backend URLs:

```text
http://127.0.0.1:8000/health
http://127.0.0.1:8000/docs
```

---

## Run the Frontend

Open a second terminal:

```powershell
cd C:\medical-ai-bot\frontend
npm run dev
```

Frontend URL:

```text
http://localhost:3000
```

---

## Frontend Test Cases

### Test 1: Billing SQL RAG

Login as:

```text
Billing Ravi - Billing Executive
```

Ask:

```text
How many claims are rejected?
```

Expected:

```text
There are 12 rejected claims.
Retrieval type: SQL RAG
No document sources returned.
```

### Test 2: Nurse SQL Access Block

Login as:

```text
Nurse Priya - Nurse
```

Ask:

```text
How many claims are rejected?
```

Expected:

```text
Access Blocked
As a nurse, you do not have access to SQL analytics.
```

### Test 3: Doctor Clinical Document RAG

Login as:

```text
Dr. Mehta - Doctor
```

Ask:

```text
What drug formulary guidance is available for antibiotic use?
```

Expected:

```text
Retrieval type: Hybrid RAG
Sources from clinical or nursing collections.
```

### Test 4: Nurse Nursing Document RAG

Login as:

```text
Nurse Priya - Nurse
```

Ask:

```text
What infection control guidance should nurses follow?
```

Expected:

```text
Retrieval type: Hybrid RAG
Sources from nursing or general collections only.
```

### Test 5: Technician Equipment Document RAG

Login as:

```text
Tech Anand - Technician
```

Ask:

```text
What equipment maintenance guidance is available?
```

Expected:

```text
Retrieval type: Hybrid RAG
Sources from equipment or general collections only.
```

## Security & Access Control

- Secrets are loaded from environment variables.
- `.env` and `.env.local` must not be committed.
- SQL execution is restricted to read-only `SELECT` statements.
- SQL analytics is restricted to `billing_executive` and `admin`.
- Document retrieval is restricted through Qdrant metadata filters.
- Frontend role display is informational only.
- Access control is enforced on the backend.
- Source citations are returned with document answers.
- Generated answers are grounded in retrieved context.

## Security & Access Control

Medical-AI Bot is designed with backend-enforced access control.

- Secrets and API keys are loaded from environment variables.
- `.env` files are excluded from version control.
- SQL execution is restricted to read-only `SELECT` statements.
- SQL analytics is available only to `billing_executive` and `admin`.
- Document access is enforced through metadata filters at the Qdrant retrieval layer.
- The frontend displays the user role and accessible collections, but it does not decide permissions.
- Source citations are returned with document answers for traceability.

## Known Limitations

- Demo authentication is intentionally simple and designed for assignment/demo usage.
- The frontend currently sends the access token in the request body instead of using an authorization header.
- Qdrant is configured in local mode for easy development setup.
- SQL RAG is limited to read-only analytics over the provided SQLite database.
- The routing logic uses rule-based detection to decide between SQL analytics and document RAG.
- The system is not intended for real patient-care decisions without clinical validation, audit controls, and production-grade security review.

---

## Future Improvements

Potential improvements include:

- Replace demo authentication with production-grade OAuth or enterprise identity provider integration.
- Move access token handling to authorization headers.
- Add persistent user/session management.
- Add automated API tests with pytest.
- Add frontend component tests.
- Add Docker support for easier deployment.
- Add Qdrant server/cloud deployment option.
- Add observability for retrieval latency, reranking quality, and blocked-access events.
- Add admin-facing audit logs for sensitive access attempts.
- Add stricter SQL query normalization and validation for generated queries.

---

## Repository Hygiene

The repository excludes generated and sensitive files such as:

```text
.env
frontend/.env.local
.venv/
__pycache__/
frontend/node_modules/
frontend/.next/
data/vector_store/
```

Shareable configuration is provided through:

```text
.env.example
```

---

## Author

**Abhishek Chauhan**

This project demonstrates a secure healthcare RAG system with role-based document retrieval, SQL analytics, FastAPI backend integration, and a Next.js frontend.

---

## Project Status

Medical-AI Bot is functionally complete for portfolio and assignment review.

Completed capabilities include:

- Role-aware login
- Backend-enforced access control
- Docling-based document processing
- Hybrid dense and sparse retrieval
- Retrieval-layer RBAC filtering
- Cross-encoder reranking
- SQL RAG over SQLite
- Gemini-based answer generation
- FastAPI backend
- Next.js frontend
- Source citations
- Blocked-access responses
