import pickle
import numpy as np
import faiss

from sentence_transformers import SentenceTransformer

# =========================================================
# LOAD INDEXES + DATA
# =========================================================

index = faiss.read_index("catalog.index")

with open("catalog_data.pkl", "rb") as f:
    data = pickle.load(f)

catalog = data["catalog"]
bm25 = data["bm25"]

model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

# =========================================================
# QUERY EXPANSION HINTS
# Learned from 10 sample traces — maps trigger words
# to semantically related terms in the catalog
# =========================================================

QUERY_HINTS = {

    # ---- LEADERSHIP / EXECUTIVE ----
    "leadership": [
        "executive", "manager", "leadership", "strategic",
        "director", "cxo", "opq leadership report",
    ],
    "cxo": [
        "executive", "director", "leadership",
        "occupational personality questionnaire opq32r",
        "opq universal competency report", "opq leadership report",
    ],
    "executive": [
        "leadership", "strategic", "director",
        "occupational personality questionnaire opq32r",
    ],

    # ---- PERSONALITY ----
    "personality": [
        "behavior", "opq", "personality",
        "occupational personality questionnaire",
    ],

    # ---- GRADUATE ----
    "graduate": [
        "entry-level", "campus", "graduate scenarios",
        "early career", "verify interactive g",
    ],
    "trainee": [
        "graduate", "entry-level", "graduate scenarios",
        "verify interactive g",
    ],

    # ---- SALES ----
    "sales": [
        "sales", "client", "customer", "commercial",
        "opq mq sales report", "sales transformation",
        "motivation questionnaire",
    ],
    "reskill": [
        "global skills assessment", "global skills development report",
        "competencies", "development 360",
    ],
    "talent audit": [
        "global skills assessment", "global skills development report",
        "competencies",
    ],

    # ---- NUMERICAL / COGNITIVE ----
    "numerical": [
        "numerical", "quantitative", "reasoning", "arithmetic",
        "verify numerical",
    ],
    "cognitive": [
        "ability", "aptitude", "reasoning", "verbal", "inductive",
        "verify interactive g",
    ],
    "reasoning": [
        "verify", "ability", "aptitude", "inductive",
        "numerical reasoning", "verbal reasoning",
    ],

    # ---- CONTACT CENTER / CUSTOMER SERVICE ----
    "contact center": [
        "svar spoken english", "contact center call simulation",
        "customer service", "entry level customer serv",
    ],
    "contact centre": [
        "svar spoken english", "contact center call simulation",
        "customer service",
    ],
    "call centre": [
        "svar", "spoken english", "call simulation",
        "customer service phone simulation",
    ],
    "customer service": [
        "svar", "contact center", "customer service phone simulation",
        "entry level customer serv retail",
    ],
    "inbound": [
        "svar spoken english", "contact center call simulation",
        "customer service",
    ],

    # ---- SAFETY / INDUSTRIAL ----
    "safety": [
        "dependability", "dsi", "dependability and safety instrument",
        "workplace health and safety", "safety and dependability",
    ],
    "chemical": [
        "safety", "dependability", "workplace health",
        "dependability and safety instrument",
    ],
    "plant operator": [
        "safety", "dependability", "frontline",
        "manufac indust safety", "dsi",
    ],
    "industrial": [
        "manufacturing", "safety dependability 8.0",
        "workplace health and safety",
    ],
    "frontline": [
        "entry level", "contact center", "retail",
        "dependability", "safety",
    ],

    # ---- ADMIN / OFFICE ----
    "admin": [
        "excel", "word", "microsoft", "office",
        "ms excel", "ms word", "administrative",
    ],
    "excel": [
        "microsoft excel", "spreadsheet", "ms excel",
        "microsoft excel 365",
    ],
    "word": [
        "microsoft word", "ms word", "document",
        "microsoft word 365",
    ],
    "administrative": [
        "ms excel", "ms word", "microsoft excel 365",
        "microsoft word 365",
    ],

    # ---- HEALTHCARE ----
    "healthcare": [
        "hipaa", "medical terminology", "clinical",
        "dependability and safety instrument",
    ],
    "medical": [
        "hipaa", "medical terminology", "healthcare",
        "dependability and safety instrument dsi",
    ],
    "hipaa": [
        "hipaa security", "medical terminology",
        "dependability and safety instrument",
    ],

    # ---- TECH / ENGINEERING ----
    "aws": ["amazon web services", "cloud", "deployment"],
    "docker": ["container", "devops", "docker"],
    "spring": ["java", "spring framework", "backend"],
    "java": ["core java", "spring", "java advanced"],
    "sql": ["sql", "database", "relational"],
    "rust": [
        "systems programming", "networking", "linux",
        "smart interview live coding", "linux programming",
    ],
    "networking": [
        "networking implementation", "linux programming",
        "smart interview live coding",
    ],
    "linux": [
        "linux programming", "networking implementation",
        "systems programming",
    ],
    "full stack": [
        "core java", "spring", "sql", "angular",
        "verify interactive g", "occupational personality questionnaire",
    ],
    "senior engineer": [
        "core java advanced", "verify interactive g",
        "occupational personality questionnaire opq32r",
    ],

    # ---- ABBREVIATION EXPANSIONS ----
    "gsa": [
        "global skills assessment",
        "global skills development report",
        "great 8 domains",
    ],
    "opq": [
        "occupational personality questionnaire",
        "opq32r", "opq32", "personality behavior",
    ],
    "sjt": [
        "situational judgment", "biodata",
        "scenarios", "work context decision",
    ],
    "mq": [
        "motivation questionnaire", "drives", "motivation",
    ],
    "verify": [
        "numerical reasoning", "verbal reasoning",
        "inductive reasoning", "verify interactive",
    ],
    "dsi": [
        "dependability and safety instrument",
        "reliability", "integrity",
    ],
    "svar": [
        "spoken english", "voice assessment",
        "contact center", "english proficiency",
    ],
}

# =========================================================
# NAMED ASSESSMENT ALIASES
# Direct name/abbreviation → full catalog name expansion
# =========================================================

KNOWN_ALIASES = {
    "opq":         "occupational personality questionnaire opq32r",
    "opq32r":      "occupational personality questionnaire opq32r",
    "gsa":         "global skills assessment global skills development report",
    "mq":          "motivation questionnaire",
    "verify":      "shl verify interactive",
    "verify g+":   "shl verify interactive g",
    "sjt":         "situational judgment graduate scenarios",
    "dsi":         "dependability and safety instrument",
    "svar":        "svar spoken english",
    "ucr":         "opq universal competency report",
    "g+":          "shl verify interactive g",
}


def expand_query(query: str) -> str:
    query_lower = query.lower()
    expanded_terms = []
    for trigger, additions in QUERY_HINTS.items():
        if trigger in query_lower:
            expanded_terms.extend(additions)
    return (query + " " + " ".join(expanded_terms)).strip()


def extract_named_assessments(query: str) -> list[str]:
    found = []
    q = query.lower()
    for alias, full_name in KNOWN_ALIASES.items():
        if alias in q:
            found.append(full_name)
    return found


# =========================================================
# HEURISTIC BOOSTING
# All boost values learned from trace patterns
# =========================================================

def apply_boosts(combined: float, item: dict, query_lower: str) -> float:

    name        = item["name"].lower()
    description = item.get("description", "").lower()
    keys        = " ".join(item.get("keys", [])).lower()
    job_levels  = " ".join(item.get("job_levels", [])).lower()
    combined_text = f"{name} {description} {keys} {job_levels}"

    # ---- LEADERSHIP ----
    if any(w in query_lower for w in ["leadership", "cxo", "executive", "director"]):
        if "opq32r" in name or "occupational personality questionnaire opq32r" in name:
            combined += 0.30
        if "opq leadership report" in name:
            combined += 0.25
        if "opq universal competency report" in name:
            combined += 0.20
        if "leadership" in combined_text and "opq" in combined_text:
            combined += 0.10

    # ---- PERSONALITY / OPQ instrument priority ----
    if "personality" in query_lower or "opq" in query_lower:
        if "opq32r" in name or "occupational personality questionnaire opq32r" in name:
            combined += 0.30
        elif "opq" in name and "report" not in name:
            combined += 0.15
        elif "opq" in combined_text and "report" not in name:
            combined += 0.08

    # ---- GSA ----
    if "gsa" in query_lower or "global skills" in query_lower:
        if "global skills assessment" in combined_text and "report" not in name:
            combined += 0.30
        if "global skills development report" in name:
            combined += 0.20

    # ---- SALES ----
    if "sales" in query_lower:
        if "opq mq sales report" in name:
            combined += 0.20
        if "sales transformation" in name:
            combined += 0.18
        if "global skills assessment" in combined_text:
            combined += 0.10

    # ---- GRADUATE ----
    if "graduate" in query_lower or "trainee" in query_lower:
        if "graduate scenarios" in name:
            combined += 0.25
        if "verify interactive g" in name:
            combined += 0.15

    # ---- CONTACT CENTER / CUSTOMER SERVICE ----
    if any(w in query_lower for w in ["contact center", "contact centre", "call centre",
                                       "customer service", "inbound"]):
        if "svar spoken english" in name:
            combined += 0.30
        if "contact center call simulation" in name:
            combined += 0.28
        if "customer service" in name:
            combined += 0.20
        if "entry level customer serv" in name:
            combined += 0.22

    # ---- SAFETY / INDUSTRIAL / CHEMICAL ----
    if any(w in query_lower for w in ["safety", "chemical", "plant", "industrial",
                                       "dependab", "dsi"]):
        if "dependability and safety instrument" in name:
            combined += 0.30
        if "safety and dependability" in name or "safety & dependability" in name:
            combined += 0.28
        if "workplace health and safety" in name:
            combined += 0.20

    # ---- ADMIN / OFFICE ----
    if any(w in query_lower for w in ["admin", "excel", "word", "office", "administrative"]):
        if "microsoft excel 365" in name or "microsoft word 365" in name:
            combined += 0.25
        if "ms excel" in name or "ms word" in name:
            combined += 0.20

    # ---- HEALTHCARE / MEDICAL ----
    if any(w in query_lower for w in ["healthcare", "medical", "hipaa", "clinic"]):
        if "hipaa" in name:
            combined += 0.30
        if "medical terminology" in name:
            combined += 0.25
        if "dependability and safety instrument" in name:
            combined += 0.15

    # ---- TECH ROLES ----
    if "numerical" in query_lower:
        if "numerical" in combined_text:
            combined += 0.12

    if "cognitive" in query_lower:
        if "ability" in combined_text:
            combined += 0.10
        if "reasoning" in combined_text:
            combined += 0.10

    if "rust" in query_lower or "networking" in query_lower:
        if "smart interview live coding" in name:
            combined += 0.25
        if "linux programming" in name:
            combined += 0.20
        if "networking and implementation" in name:
            combined += 0.20

    if "java" in query_lower:
        if "core java" in name:
            combined += 0.25
        if "spring" in name:
            combined += 0.15

    if "aws" in query_lower:
        if "amazon web services" in name:
            combined += 0.35

    if "docker" in query_lower:
        if "docker" in name:
            combined += 0.35

    if "sql" in query_lower:
        if "sql" in name and "nosql" not in name:
            combined += 0.30

    # ---- DSI explicit ----
    if "dsi" in query_lower:
        if "dependability and safety instrument" in name:
            combined += 0.40

    # ---- SVAR explicit ----
    if "svar" in query_lower:
        if "svar spoken english" in name:
            combined += 0.40

    return combined


# =========================================================
# MAIN HYBRID SEARCH
# =========================================================

def hybrid_search(query: str, top_k: int = 10) -> list[dict]:

    # Named entity expansion first
    named = extract_named_assessments(query)
    if named:
        query = query + " " + " ".join(named)

    expanded_query = expand_query(query)
    query_lower = expanded_query.lower()

    # ---- SEMANTIC SEARCH ----
    q_emb = model.encode(
        [expanded_query],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    scores, indices = index.search(q_emb, top_k * 3)

    semantic_scores = {}
    for score, idx in zip(scores[0], indices[0]):
        if idx >= 0:
            semantic_scores[idx] = float(score)

    # ---- BM25 SEARCH ----
    tokens = expanded_query.lower().split()
    bm25_scores_raw = bm25.get_scores(tokens)
    max_bm25 = max(bm25_scores_raw) if max(bm25_scores_raw) > 0 else 1
    bm25_scores = {
        i: float(s / max_bm25)
        for i, s in enumerate(bm25_scores_raw)
    }

    # ---- CANDIDATE POOL ----
    candidate_indices = set(semantic_scores.keys()) | {
        i for i, s in bm25_scores.items() if s > 0.10
    }

    combined_scores = {}
    for idx in candidate_indices:
        semantic  = semantic_scores.get(idx, 0.0)
        bm25_score = bm25_scores.get(idx, 0.0)
        combined  = (0.65 * semantic) + (0.35 * bm25_score)
        combined  = apply_boosts(combined, catalog[idx], query_lower)
        combined_scores[idx] = combined

    # ---- SORT ----
    sorted_indices = sorted(
        combined_scores,
        key=lambda i: combined_scores[i],
        reverse=True
    )

    # ---- DIVERSITY FILTERING ----
    results = []
    seen_names = set()

    for idx in sorted_indices:
        item = catalog[idx]
        base_name = (
            item["name"].lower()
            .replace("1.0", "").replace("2.0", "")
            .split("(")[0].strip()
        )
        if base_name in seen_names:
            continue
        seen_names.add(base_name)

        results.append({
            "name":        item["name"],
            "url":         item["link"],
            "test_type":   item["test_type"],
            "description": item.get("description", ""),
            "job_levels":  item.get("job_levels", []),
            "duration":    item.get("duration", ""),
            "keys":        item.get("keys", []),
            "score":       round(combined_scores[idx], 4),
        })

        if len(results) >= top_k:
            break

    return results
