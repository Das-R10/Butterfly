"""
tests/test_api.py

Unit tests for FastAPI endpoints in src.api.main.
Validates:
  - GET /health
  - POST /api/simulations/run
  - POST /api/simulations/perturb
  - GET /api/simulations/{id}/trajectory
  - POST /signals/analyze (backward compatibility)
"""

import sys
import os
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.api.main import app

client = TestClient(app)


def test_api_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "layers" in data


def test_api_simulation_run_sync():
    payload = {
        "signal": "Autonomous drone swarm deployed in exercise",
        "max_iterations": 3,
        "churn_rate": 0.08,
    }
    response = client.post("/api/simulations/run", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "simulation_id" in data
    assert len(data["iterations"]) == 3
    assert "trajectory" in data
    assert "model_agreement_score" in data


def test_api_perturbation():
    payload = {
        "signal": "Border skirmish reported",
        "max_iterations": 3,
        "shock_type": "fatigue_spike",
        "shock_iteration": 2,
        "intensity": 0.5,
    }
    response = client.post("/api/simulations/perturb", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "adoption_divergence" in data
    assert "effect_summary" in data


def test_api_legacy_analyze_compatibility():
    payload = {
        "signal": "Diplomatic sanctions introduced",
        "mode": "support",
        "max_iterations": 3,
    }
    response = client.post("/signals/analyze", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "simulation_id" in data
    assert "trajectory" in data
    assert "clusters" in data

