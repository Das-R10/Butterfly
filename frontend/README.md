# The Butterfly Algorithm — Frontend Research Interface

A high-performance computational social-science research dashboard built with **React**, **TypeScript**, **Vite**, **Tailwind CSS**, **Recharts**, and **React Flow** (@xyflow/react), integrating seamlessly with The Butterfly Algorithm's FastAPI 4-layer behavioral simulation engine.

---

## 🔬 Core 4-Layer Architecture

1. **Micro Layer (Cognitive Archetypes)**: 5 distinct archetypes evaluated independently across 2 LLM engines (10 validated structured interpretations per iteration: stance, virality, credibility, emotional charge).
2. **Meso Layer (Influence Network)**: 8-node weighted directed influence graph separating broadcast media pressure from peer-to-peer social contagion without double-counting.
3. **Macro Layer (Modified Bass Diffusion)**: Deterministic differential diffusion modeling adoption trajectory $N(t)$ with adoption fatigue/churn parameter $c$:
   $$\frac{dN}{dt} = \left[ p + q \cdot \frac{N}{M} \right] (M - N) - c \cdot N$$
   Strictly distinguishes conversational buzz $S(t)$ from persistent active adoption $N(t)$.
4. **Feedback Layer (Iterative Recurrence)**: Aggregated macro state, polarization index, and dominant framing injected into the context of subsequent rounds (bounded strictly to 3–5 iterations).

---

## 🚀 Quickstart Guide

### 1. Start the FastAPI Backend
From the workspace root (`ba/`):
```bash
# Terminal 1: Launch FastAPI backend on port 8000
python -m uvicorn src.api.main:app --app-dir ba --reload --port 8000
```
Backend API will be accessible at:
- Heartbeat: `http://localhost:8000/health`
- Swagger Docs: `http://localhost:8000/docs`
- WebSocket progress: `ws://localhost:8000/ws/{job_id}`

### 2. Start the Frontend Development Server
From the `frontend/` directory:
```bash
# Terminal 2: Launch Vite development server
cd frontend
npm run dev
```
Open your browser at:
`http://localhost:5173`

To create an optimized production build:
```bash
npm run build
npm run preview
```

---

## 📑 Page Directory & Capabilities

| Page | Route | Description |
|---|---|---|
| **Overview** | `/` | Platform architecture explanation, platform KPIs, recent simulation history table, and quick launch action. |
| **New Simulation** | `/new` | Exogenous signal entry, quick benchmark templates, dual LLM provider selection (Mock, Groq, Gemini, Ollama), 3–5 iteration slider, and macro parameters ($M$, $c$, $\alpha$). |
| **Live Simulation** | `/live` | Real-time WebSocket execution progress streaming with fallback polling, live execution log terminal, tabbed inspection across Micro, Meso, Macro, and Feedback layers, and discrete cycle selector. |
| **Narrative Analysis** | `/narrative` | Deep inspection of all 10 interpretations per cycle, side-by-side Model A vs Model B stance differences, narrative keyword tags, and semantic convergence metrics. |
| **Influence Network** | `/network` | Interactive React Flow canvas with custom nodes (Archetypes, Broadcast, Early Adopters, Public), animated influence edges, node activation inspector, and channel breakdown. |
| **Macro Trajectory** | `/trajectory` | Interactive Recharts area/line curves showing Cumulative Adoption, Active Adoption ($N(t)$), Attention Indicator ($S(t)$), and Velocity ($dN/dt$). Explicit labeling distinguishing attention from adoption. |
| **Simulation History** | `/history` | Searchable and filterable archive of past simulation sessions persisted in `data/sessions/`, with one-click inspection loading into active context. |
| **Experiment Lab** | `/lab` | Counterfactual shock perturbation experiment interface (debunking injection, influencer blast, fatigue spike) with dual-trajectory divergence curves; plus empirical evaluation against held-out observations (MSE, MAE, directional accuracy). |
| **Settings / Diagnostics** | `/settings` | Backend heartbeat check, round-trip latency, LLM provider guides, environment variable configurations, and local browser cache reset. |

---

## 🛡️ Scientific Methodology: Attention vs Adoption

A fundamental rule enforced throughout this interface:
- **Conversational Attention $S(t)$** represents media buzz, discourse volume, and virality.
- **Active Adoption $N(t)$** represents persistent converted users governed by Bass innovation, imitation, and churn decay.
- **Attention volume is NEVER labeled or reported as actual adoption.**
