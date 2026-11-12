# The Butterfly Algorithm (4-Layer Behavioral Dynamics Simulation Engine)

**The Butterfly Algorithm** is a computational social-science framework modeling how micro-level human behavioral signals propagate across social networks, produce macroscopic trend trajectories, and loop back via iterative feedback dynamics.

Built for 7th semester CSE minor project.

---

## 🏛️ Architecture Overview

```
                      ┌───────────────────────────────────────┐
                      │            EXOGENOUS SIGNAL           │
                      └──────────────────┬────────────────────┘
                                         │
                                         ▼
  ┌───────────────────────────────────────────────────────────────────────────┐
  │ 1. MICRO LAYER                                                            │
  │    5 Cognitive Archetypes × 2 Independent LLMs = 10 Interpretations/Round   │
  │    • Institutional Analyst  • Skeptic Debunker  • Pattern Seeker          │
  │    • Doomer                 • Amplifier Influencer                        │
  │    (Stance: [-1, +1], Virality: [0, 1], Credibility: [0, 1], Emotion)      │
  └──────────────────────────────────────┬────────────────────────────────────┘
                                         │
                                         ▼
  ┌───────────────────────────────────────────────────────────────────────────┐
  │ 2. MESO LAYER (INFLUENCE NETWORK)                                         │
  │    8-Node Weighted Directed Graph                                         │
  │    • External broadcast influence (Media/Influencers)                     │
  │    • Internal peer-to-peer contagion without double-counting              │
  └──────────────────────────────────────┬────────────────────────────────────┘
                                         │
                                         ▼
  ┌───────────────────────────────────────────────────────────────────────────┐
  │ 3. MACRO LAYER (MODIFIED BASS DIFFUSION)                                  │
  │    dN/dt = [p + q*(N/M)] * (M - N) - c*N                                  │
  │    • Adoption Fatigue / Churn Parameter (c)                               │
  │    • Strict separation: Attention Buzz S(t) vs Active Adoption N(t)       │
  └──────────────────────────────────────┬────────────────────────────────────┘
                                         │
                                         ▼
  ┌───────────────────────────────────────────────────────────────────────────┐
  │ 4. FEEDBACK LAYER (ITERATIVE RECURRENCE)                                  │
  │    Macro state & polarization injected into next iteration context        │
  │    Strictly bounded to 3–5 iterations                                     │
  └───────────────────────────────────────────────────────────────────────────┘
```

---

## 💻 Tech Stack

- **Backend**: Python 3.11, FastAPI, Pydantic v2, WebSockets, NumPy, Loguru, Uvicorn.
- **Frontend**: React 19, TypeScript, Vite, Tailwind CSS, Recharts, React Flow (`@xyflow/react`), Lucide React.
- **LLM Providers**: Groq (`llama-3.3-70b-versatile`), Google Gemini (`gemini-1.5-flash`), Ollama (`llama3`), Deterministic Mock (offline/zero-dependency).

---

## 🚀 Running the Project

### 1. Prerequisites & Environment
```bash
# Install backend requirements
pip install -r ba/requirements.txt
pip install loguru vaderSentiment

# Copy environment template if using Cloud LLMs
copy .env.example .env
```

### 2. Start FastAPI Backend (Port 8000)
```bash
# From workspace root (D:\ba):
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
# Or with explicit --app-dir:
python -m uvicorn src.api.main:app --app-dir ba --host 0.0.0.0 --port 8000 --reload
```
Backend runs at `http://localhost:8000` (Docs at `http://localhost:8000/docs`).

### 3. Start Frontend Dashboard (Port 5173)
```bash
cd frontend
npm install
npm run dev
```
Frontend runs at `http://localhost:5173`.

---

## 🦙 Local Ollama Setup (Recommended)

1. **Install Ollama** (Windows 11):
   ```powershell
   winget install Ollama.Ollama
   ```
2. **Pull the Dual Models**:
   ```powershell
   # Model A (Primary Analyst)
   ollama pull qwen3.5:4b

   # Model B (Independent Reviewer)
   ollama pull gemma3:4b
   ```

---

## 🧪 Running Automated Tests
```bash
# Run complete test suite (25 passed)
pytest
```
All 25 core engine, REST/WebSocket, hardware diagnostics, and Ollama integration tests pass deterministically.

