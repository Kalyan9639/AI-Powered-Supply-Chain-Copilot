# Agentic AI Supply Chain Copilot

An AI-powered agent that monitors Indian logistics news, checks it against your business profile and past incidents, and proposes practical actions using Gemma via Hugging Face.

It also includes a browser-based runtime manager at `/manager` so you can update key environment values from any device, including your phone.

## What This Project Does

The copilot runs in a loop:

1. Scrape logistics news from the configured source pages.
2. Convert each article into embeddings for duplicate detection and memory search.
3. Store the article in ChromaDB if it is new.
4. Send the article, your business profile, and similar incidents to Gemma.
5. Parse Gemma's answer into a structured risk assessment.
6. Save the assessment and expose it through the API and frontend.

## Data Flow

```text
ITLN news pages
    -> Crawl4AI scraper
    -> NewsArticle object
    -> SentenceTransformer embeddings
    -> ChromaDB news memory
    -> Similar incident lookup
    -> Gemma on Hugging Face
    -> RiskAssessment object
    -> ChromaDB assessment memory
    -> FastAPI endpoints
    -> Frontend dashboard
```

### Storage and Config Locations

- `data/config/business_profile.json` is the primary business profile used by the agent.
- `config.json` is also supported as a fallback business profile source.
- `data/config/runtime.env` stores runtime overrides edited from `manager.html`.
- `.env` remains the fallback source for local development and first-time setup.
- `data/cache/chromadb` stores the local ChromaDB persistence files.
- `data/cache` is also used as the local cache base for Crawl4AI.

## Project Structure

```text
src/
  api/       FastAPI backend
  agent/     Agent orchestration and Gemma integration
  scraper/   Crawl4AI news scraping
  memory/    ChromaDB vector store
  models/    Pydantic schemas
  utils/     Logging and embeddings
  main.py    CLI entry point
frontend/    Web dashboard
data/        Business profile and local storage
```

## Requirements

- Python 3.11+
- Hugging Face API key in `.env`
- A virtual environment with the project dependencies installed
- Docker if you want the Cloud Run/container workflow locally

## Environment Setup

Copy `.env.example` to `.env` and set your Hugging Face key:

```env
HF_API_KEY=your-huggingface-api-key
SCRAPE_INTERVAL_MINUTES=15
LOG_LEVEL=INFO
DEBUG=False
```

You can also update these runtime values after the app is running by opening:

- `http://localhost:8000/manager`

The manager page writes the values into `data/config/runtime.env`, and the backend reloads them without requiring a restart.

## Install Dependencies

Using `uv`:

```bash
uv sync
```

Using `pip`:

```bash
pip install -r requirements.txt
```

## Ways to Run the Project

### 1. Cloud Run friendly local server

This is the same HTTP app you will deploy in Cloud Run:

```bash
uvicorn api.main:app --app-dir src --host 0.0.0.0 --port 8000
```

Then open:

- `http://localhost:8000/` for the dashboard
- `http://localhost:8000/manager` for the runtime manager
- `http://localhost:8000/health` for a health check
- `http://localhost:8000/api/run-cycle` to trigger a cycle manually

### 2. Run one agent cycle

Use this for a quick test of scraping, memory lookup, and Gemma inference:

```bash
python src/main.py --mode single
```

### 3. Run the continuous scheduler locally

Use this only for local development, not for the Cloud Run container:

```bash
python src/main.py --mode scheduler
```

## API Endpoints

- `GET /health` - service health check
- `POST /api/run-cycle` - run one full agent cycle
- `GET /api/assessments` - list recent risk assessments
- `GET /api/assessments/{news_id}` - fetch a specific assessment
- `GET /api/news` - placeholder endpoint for recent news output
- `GET /api/runtime-config` - read the live runtime values used by the app
- `PUT /api/runtime-config` - update the live runtime values from the manager page
- `GET /manager` - open the runtime manager UI

## How the Agent Thinks

The agent uses:

- Gemma as the reasoning model
- local embeddings for semantic search
- ChromaDB for short-term and historical memory
- a business profile file so decisions stay specific to your operation

For each article, the prompt includes:

- your business name
- your commodity
- your region
- your business rules
- the news article content
- similar incidents from memory

Gemma returns a structured result:

- `risk_level`
- `reasoning`
- `proposed_action`
- `confidence`

## Tech Stack

- **LLM**: `google/gemma-4-26B-A4B-it` via `huggingface_hub`
- **Embeddings**: `sentence-transformers`
- **Vector DB**: ChromaDB
- **Scraping**: Crawl4AI
- **Backend**: FastAPI
- **Validation**: Pydantic

## Notes

- The app defaults to the business profile at `data/config/business_profile.json`.
- Runtime config edits are stored in `data/config/runtime.env` locally, or in the Cloud Run writable storage directory if `APP_STORAGE_DIR` is set.
- Business profile edits from the dashboard are also file-backed, so they follow the same Cloud Run persistence limitations.
- ChromaDB and Crawl4AI both use the `data/cache` subtree so local state stays inside the project.
- If Hugging Face requests fail, check that the API key in `.env` is valid and that your account can access the chosen Gemma model.
- On Cloud Run, the container filesystem is ephemeral. The runtime manager works, but changes are only durable while the instance exists unless you wire the config into a persistent backend such as Firestore or Secret Manager.
- The Docker image ships with Playwright and Chromium so Crawl4AI can scrape pages that require a real browser.
- For scheduled scraping in Cloud Run, keep the service request-driven and use Cloud Scheduler or a manual call to `POST /api/run-cycle` instead of an always-on background loop.

## Docker Build

Build the container locally:

```bash
docker build -t agentic-copilot .
```

Run it locally:

```bash
docker run --rm -p 8080:8080 \
  -e HF_API_KEY=your-huggingface-api-key \
  agentic-copilot
```

Cloud Run will use the same image and the same `uvicorn api.main:app` entrypoint.

## License

MIT License
