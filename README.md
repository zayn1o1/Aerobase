# Airport AI Management System

This project has three parts:

- `frontend/` contains the HTML dashboards.
- `backend/` contains the Python FastAPI AI service.
- Supabase stores authentication, airport data, flights, delays, gates, and recommendations.

## Project Structure

```text
ai/
  frontend/
    airport_auth.html
    staff_dashboard.html

  backend/
    main.py
    train_model.py
    requirements.txt
    .env
    .env.example

  README.md
  .gitignore
```

## Setup

Use the existing Desktop virtual environment:

```bash
source ~/Desktop/.venv/bin/activate
pip install -r backend/requirements.txt
```

Create `backend/.env`:

```env
SUPABASE_URL=https://wsdbhfxucobeqfhjcggk.supabase.co
SUPABASE_SERVICE_ROLE_KEY=your-service-role-key
MODEL_PATH=backend/model.pkl
```

The service role key must stay in `backend/.env`. Do not put it in the frontend HTML.

## Train The Delay Model

Run this after `backend/.env` is ready:

```bash
source ~/Desktop/.venv/bin/activate
python backend/train_model.py
```

This creates:

```text
backend/model.pkl
```

If the training script warns about too few delayed or on-time examples, add more historical flight rows or delay logs in Supabase.

## Run Locally

Open two terminals.

Terminal 1: Python AI backend

```bash
cd ~/Desktop/ai\ 
source ~/Desktop/.venv/bin/activate
uvicorn backend.main:app --host 127.0.0.1 --port 8010
```

Check:

```text
http://127.0.0.1:8010/health
```

Terminal 2: frontend server

```bash
cd ~/Desktop/ai\ 
python3 -m http.server 5500 -d frontend
```

Open:

```text
http://127.0.0.1:5500/airport_auth.html?logout=1
```

Do not open the HTML files directly with `file://...`; Supabase auth sessions are more reliable through the local server URL.

## Deploy Frontend On Vercel

This repo includes `vercel.json`, so Vercel can serve the files from `frontend/`.

Deploy the project root to Vercel. Then open:

```text
https://your-vercel-app.vercel.app/
```

The Python backend is not deployed by Vercel in this setup. For local development, keep running:

```bash
source ~/Desktop/.venv/bin/activate
uvicorn backend.main:app --host 127.0.0.1 --port 8010
```

If the deployed Vercel frontend cannot call `http://localhost:8010`, deploy the Python backend separately and update `AI_API_BASE` in `frontend/staff_dashboard.html`.

## AI Features

The staff dashboard calls the Python backend for:

- A* gate assignment recommendations
- ML-assisted alternative flight recommendations
- Delay prediction through the trained Random Forest model

The Python backend writes generated recommendations into the Supabase `recommendations` table. The staff dashboard then reloads and displays them.
