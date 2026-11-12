"""
src/ingestion/entity_extractor.py

Extracts entities and keywords from signal text using the local
Ollama model — no spaCy, no new dependencies, no version conflicts.

Uses the same Mistral 7B instance that's already running for
the personality engine. Returns structured JSON for dynamic search.
"""

import json
import re
import requests
from loguru import logger

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "mistral:7b"

EXTRACTION_PROMPT = """You are a geopolitical signal analyst. Extract structured information from the following news signal.

Return ONLY valid JSON with this exact structure — no explanation, no markdown, no extra text:
{{
  "entities": {{
    "countries": [],
    "organizations": [],
    "people": [],
    "locations": []
  }},
  "keywords": [],
  "topic": "",
  "signal_type": ""
}}

Rules:
- countries: list of country/nation names mentioned
- organizations: NATO, UN, WEF, company names, military bodies etc
- people: named individuals (politicians, leaders, officials)
- locations: specific places, straits, cities, bases, regions
- keywords: 5-8 most important words/phrases for searching this story
- topic: one of [conflict, economics, trade, diplomacy, energy, politics, security, other]
- signal_type: one of [breaking_news, analysis, rumor, official_statement, social_media]

Signal:
{signal}

JSON:"""


def _call_ollama(prompt: str) -> str | None:
    """Call local Ollama API and return raw text response."""
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,   # low temp for structured extraction
                    "num_predict": 400,
                }
            },
            timeout=30,
        )
        if resp.status_code != 200:
            logger.warning(f"Ollama returned {resp.status_code}")
            return None
        return resp.json().get("response", "").strip()
    except Exception as e:
        logger.warning(f"Ollama call failed: {e}")
        return None


def _parse_json_response(raw: str) -> dict | None:
    """
    Extract JSON from Ollama response.
    Handles cases where the model adds extra text around the JSON.
    """
    if not raw:
        return None

    # Try direct parse first
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Try to extract JSON block
    match = re.search(r'\{.*\}', raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    logger.warning(f"Could not parse JSON from Ollama response: {raw[:100]}")
    return None


def _regex_fallback(text: str) -> dict:
    """
    Fast regex fallback when Ollama is unavailable.
    Good enough for countries, major orgs, and money values.
    """
    countries = re.findall(
        r'\b(US|USA|United States|China|Russia|Iran|Ukraine|Israel|NATO|EU|UK|'
        r'India|Pakistan|Taiwan|North Korea|Saudi Arabia|Germany|France|Japan|'
        r'Brazil|Turkey|Syria|Gaza|Lebanon|Yemen|Britain|Canada|Australia)\b',
        text, re.IGNORECASE
    )
    orgs = re.findall(
        r'\b(NATO|WEF|IMF|UN|WHO|WTO|Fed|Federal Reserve|Pentagon|'
        r'Apple|Nike|Amazon|Microsoft|Google|OPEC|CIA|Mossad|IEA)\b',
        text
    )
    money = re.findall(r'\$[\d,]+(?:\.\d+)?(?:/barrel|B|M|T)?', text)

    # Extract keywords from capitalized phrases
    keywords = list(set(
        countries[:4] +
        orgs[:2] +
        [m for m in money[:2]]
    ))

    return {
        "entities": {
            "countries": list(set(c.upper() for c in countries[:6])),
            "organizations": list(set(orgs[:4])),
            "people": [],
            "locations": [],
        },
        "keywords": keywords[:8],
        "topic": "other",
        "signal_type": "breaking_news",
        "method": "regex_fallback",
    }


def extract_entities(signal: dict) -> dict:
    """
    Extract entities and keywords from signal_text using Ollama.
    Falls back to regex if Ollama is unavailable.
    Updates signal in-place with extraction results.
    """
    text = signal.get("signal_text", "")
    if not text:
        return signal

    # Try LLM extraction first
    prompt = EXTRACTION_PROMPT.format(signal=text[:800])
    raw = _call_ollama(prompt)
    extracted = _parse_json_response(raw) if raw else None

    if extracted:
        signal["entities"] = extracted.get("entities", {})
        signal["keywords"] = extracted.get("keywords", [])
        signal["topic"] = extracted.get("topic", "other")
        signal["signal_type"] = extracted.get("signal_type", "unknown")
        signal["entity_method"] = "ollama"

        countries = extracted.get("entities", {}).get("countries", [])
        orgs = extracted.get("entities", {}).get("organizations", [])
        keywords = extracted.get("keywords", [])
        logger.info(
            f"LLM extraction — countries: {countries[:4]} | "
            f"orgs: {orgs[:3]} | keywords: {keywords[:5]}"
        )
    else:
        # Regex fallback
        logger.info("Ollama unavailable — using regex fallback for entity extraction")
        result = _regex_fallback(text)
        signal["entities"] = result["entities"]
        signal["keywords"] = result["keywords"]
        signal["topic"] = result["topic"]
        signal["signal_type"] = result["signal_type"]
        signal["entity_method"] = "regex_fallback"

    return signal


def entities_to_queries(signal: dict, max_queries: int = 5) -> list[str]:
    """
    Build search query strings from LLM-extracted entities and keywords.
    Combines entities intelligently rather than just listing them.
    """
    entities = signal.get("entities", {})
    keywords = signal.get("keywords", [])

    countries = entities.get("countries", [])
    orgs = entities.get("organizations", [])
    people = entities.get("people", [])
    locations = entities.get("locations", [])

    queries = []

    # Most specific: country + location pairs
    for loc in locations[:2]:
        if countries:
            queries.append(f"{countries[0]} {loc}")

    # Country + org pairs
    for org in orgs[:2]:
        if countries:
            queries.append(f"{countries[0]} {org}")

    # Top keyword phrases (most useful for GDELT)
    for kw in keywords[:3]:
        if kw not in queries:
            queries.append(kw)

    # Fallback: just the primary country
    if countries and countries[0] not in queries:
        queries.append(countries[0])

    # Deduplicate and limit
    seen = set()
    unique = []
    for q in queries:
        q_clean = q.strip()
        if q_clean.lower() not in seen and len(q_clean) > 2:
            seen.add(q_clean.lower())
            unique.append(q_clean)

    return unique[:max_queries]
