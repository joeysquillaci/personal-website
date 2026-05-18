# Options Trading Dashboard

Web-based options trading dashboard with a React frontend and FastAPI backend.  
The app tracks positions in SQLite, pulls live options data from Tradier, and includes a Gemini-powered AI assistant for strategy and position discussion.

## What This Project Does

- Track open and closed options positions in a ledger
- Calculate and monitor P/L over time
- Fetch stock quotes and options chain/contract pricing from Tradier
- Filter option chains around spot price using configurable strike range
- Ask Gemini for position-specific trading insights
- Import positions from a structured Excel sheet

## Tech Stack

- **Frontend:** React, Vite, React Router, Axios, Recharts
- **Backend:** FastAPI, Uvicorn, pandas, requests, python-dotenv
- **Data:** SQLite (`backend/databases/*.db`)
- **External APIs:** Tradier Options API, Gemini API

## App Sections

- **Dashboard**
  - Live open-position pricing snapshot
  - Running total P/L chart and table
- **Ledger**
  - Add, edit, close, and delete positions
  - Right-click row actions including "Ask AI"
  - Excel import and configurable visible columns
- **Option Chain**
  - Calls and puts tables by ticker + expiration
  - Strike filtering based on settings (`strike_range_percentage`)
- **AI Assistant**
  - Chat with Gemini (`gemini-2.5-flash`) from the backend
  - Send position context directly from Ledger
  - Save chats to local text files
- **Settings**
  - Profile and strategy preferences
  - Options chain strike-range configuration
- **Notebook**
  - Placeholder page (not yet implemented)

## Project Structure

```text
js_port-v4/
  backend/
    main.py              # FastAPI routes and API integrations
    database.py          # SQLite data access and metrics calculations
    requirements.txt
    settings.json        # persisted user settings
    databases/           # created at runtime for .db files
  frontend/
    src/
      pages/             # Dashboard, Ledger, Option Chain, AI Assistant, etc.
      contexts/          # P/L provider and refresh loops
      lib/api.js         # frontend API client
    package.json
    vite.config.js       # dev proxy /api -> localhost:8000
```

## Prerequisites

- Node.js 18+
- npm
- Python 3.10+
- Tradier API token
- Gemini API key (optional if you do not use AI Assistant)

## Environment Variables

Create `backend/.env`:

```env
TRADIER_TOKEN=your_tradier_token_here
GEMINI_API_KEY=your_gemini_api_key_here
```

Notes:
- `TRADIER_TOKEN` is required for quotes/chain pricing.
- `GEMINI_API_KEY` is required only for AI chat endpoints.
- Restart backend after editing `.env`.

## Local Setup

### 1) Backend

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Backend will run at `http://localhost:8000`.

### 2) Frontend

In a second terminal:

```bash
cd frontend
npm install
npm run dev
```

Frontend will run at `http://localhost:3000` and proxy `/api` requests to backend port `8000`.

## API Overview

### Health
- `GET /api/health`

### Positions / Databases
- `GET /api/databases`
- `GET /api/positions`
- `GET /api/positions/open`
- `GET /api/positions/{position_id}`
- `POST /api/positions`
- `PUT /api/positions/{position_id}`
- `DELETE /api/positions/{position_id}`
- `POST /api/positions/{position_id}/close`
- `POST /api/positions/{position_id}/roll`

### Market Data (Tradier)
- `GET /api/stock-quote/{ticker}`
- `GET /api/option-price/{ticker}/{expiration}/{strike}/{option_type}`
- `POST /api/option-prices/batch`
- `POST /api/optionchain`

### Import / Settings / AI
- `POST /api/import/excel`
- `GET /api/settings`
- `POST /api/settings`
- `GET /api/ai/status`
- `POST /api/ai/chat`

## Data and Calculations

The backend stores positions in SQLite and computes key metrics when positions close:

- `collateral`
- `p_l`
- `days`
- `roc`
- `anl_roc`

P/L chart history is persisted in browser local storage from the frontend context.

## Development Notes

- CORS currently allows local frontend origins (`3000`, `5173`, `5174`).
- Default DB name in frontend/backend is `positions.db`.
- Option chain UI enforces Friday expirations and highlights third-Friday expirations.

## Security Reminder

- Do not commit real API keys or tokens to version control.
- Rotate tokens immediately if they were exposed during development.

## Disclaimer

This software is for educational and research purposes only and is not financial advice.  
Trading options involves substantial risk.
