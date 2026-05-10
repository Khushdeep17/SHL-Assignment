import pickle
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

# =========================================================
# LOAD EMBEDDING MODEL
# =========================================================

model = SentenceTransformer("all-MiniLM-L6-v2")

# =========================================================
# QUERY EXPANSION HINTS
# =========================================================

QUERY_HINTS = {

    # leadership / executive
    "leadership": [
        "executive", "manager", "leadership",
        "strategic", "director", "cxo",
    ],

    # personality
    "personality": [
        "behavior", "opq", "personality",
        "occupational personality questionnaire",
    ],

    # graduate hiring
    "graduate": [
        "entry-level", "campus", "graduate", "early career",
    ],

    # sales
    "sales": [
        "sales", "client", "customer", "commercial",
        "opq mq sales report", "sales transformation",
    ],

    # numerical
    "numerical": [
        "numerical", "quantitative", "reasoning", "arithmetic",
    ],

    # cognitive
    "cognitive": [
        "ability", "aptitude", "reasoning", "verbal", "inductive",
    ],

    # contact center / customer service
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
    ],

    # safety / industrial
    "safety": [
        "dependability", "dsi", "dependability and safety instrument",
        "workplace health and safety", "safety and dependability",
    ],
    "chemical": [
        "safety", "dependability", "workplace health",
        "dependability and safety instrument",
    ],
    "industrial": [
        "manufacturing", "safety dependability 8.0",
        "workplace health and safety",
    ],

    # admin / office
    "admin": [
        "excel", "word", "microsoft", "ms excel", "ms word",
    ],
    "excel": [
        "microsoft excel", "ms excel", "microsoft excel 365",
    ],
    "word": [
        "microsoft word", "ms word", "microsoft word 365",
    ],

    # healthcare
    "healthcare": [
        "hipaa", "medical terminology", "dependability and safety instrument",
    ],
    "hipaa": [
        "hipaa security", "medical terminology",
        "dependability and safety instrument",
    ],

    # tech
    "aws": ["amazon web services", "cloud", "deployment"],
    "docker": ["container", "devops", "docker"],
    "java": ["core java", "spring", "java advanced"],
    "sql": ["sql", "database", "relational"],

    # abbreviation expansions
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
        "inductive reasoning", "verify",
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
# QUERY EXPANSION
# =========================================================

def expand_query(query: str) -> str:
    query_lower = query.lower()
    expanded_terms = []
    for trigger, additions in QUERY_HINTS.items():
        if trigger in query_lower:
            expanded_terms.extend(additions)
    return (query + " " + " ".join(expanded_terms)).strip()

# =========================================================
# NAMED ASSESSMENT EXTRACTION
# =========================================================

def extract_named_assessments(query: str) -> list[str]:
    known_aliases = {
        "opq":      "occupational personality questionnaire opq32r",
        "gsa":      "global skills assessment global skills development report",
        "mq":       "motivation questionnaire",
        "verify":   "shl verify interactive",
        "sjt":      "situational judgment graduate scenarios",
        "dsi":      "dependability and safety instrument",
        "svar":     "svar spoken english",
    }
    found = []
    q = query.lower()
    for alias, full_name in known_aliases.items():
        if alias in q:
            found.append(full_name)
    return found

# =========================================================
# MAIN HYBRID SEARCH
# =========================================================

def hybrid_search(query: str, top_k: int = 10) -> list[dict]:

    # Named entity expansion
    named_assessments = extract_named_assessments(query)
    if named_assessments:
        query = query + " " + " ".join(named_assessments)

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
        i: float(score / max_bm25)
        for i, score in enumerate(bm25_scores_raw)
    }

    # ---- CANDIDATE POOL ----
    candidate_indices = set(semantic_scores.keys()) | set(
        i for i, score in bm25_scores.items()
        if score > 0.10
    )

    combined_scores = {}

    for idx in candidate_indices:

        semantic   = semantic_scores.get(idx, 0.0)
        bm25_score = bm25_scores.get(idx, 0.0)
        combined   = (0.65 * semantic) + (0.35 * bm25_score)

        item          = catalog[idx]
        name          = item["name"].lower()
        description   = item.get("description", "").lower()
        keys          = " ".join(item.get("keys", [])).lower()
        combined_text = f"{name} {description} {keys}"

        # LEADERSHIP
        if any(w in query_lower for w in ["leadership", "cxo", "executive", "director"]):
            if "opq32r" in name or "occupational personality questionnaire opq32r" in name:
                combined += 0.30
            if "opq leadership report" in name:
                combined += 0.25
            if "opq universal competency report" in name:
                combined += 0.20

        # PERSONALITY / OPQ instrument priority
        if "personality" in query_lower or "opq" in query_lower:
            if "opq32r" in name or "occupational personality questionnaire opq32r" in name:
                combined += 0.25
            elif "opq" in name and "report" not in name:
                combined += 0.15
            elif "opq" in combined_text and "report" not in name:
                combined += 0.10

        # GSA
        if "gsa" in query_lower or "global skills" in query_lower:
            if "global skills assessment" in combined_text and "report" not in name:
                combined += 0.30
            if "global skills development report" in name:
                combined += 0.15

        # SALES
        if "sales" in query_lower:
            if "opq mq sales report" in name:
                combined += 0.20
            if "sales transformation" in name:
                combined += 0.18

        # GRADUATE
        if "graduate" in query_lower:
            if "graduate scenarios" in combined_text:
                combined += 0.20

        # CONTACT CENTER
        if any(w in query_lower for w in ["contact center", "contact centre",
                                           "call centre", "customer service", "inbound"]):
            if "svar spoken english" in name:
                combined += 0.30
            if "contact center call simulation" in name:
                combined += 0.28
            if "entry level customer serv" in name:
                combined += 0.22

        # SAFETY / INDUSTRIAL
        if any(w in query_lower for w in ["safety", "chemical", "plant", "industrial", "dsi"]):
            if "dependability and safety instrument" in name:
                combined += 0.30
            if "safety and dependability" in name or "safety & dependability" in name:
                combined += 0.28
            if "workplace health and safety" in name:
                combined += 0.20

        # ADMIN / OFFICE
        if any(w in query_lower for w in ["admin", "excel", "word", "administrative"]):
            if "microsoft excel 365" in name or "microsoft word 365" in name:
                combined += 0.25
            if "ms excel" in name or "ms word" in name:
                combined += 0.20

        # HEALTHCARE
        if any(w in query_lower for w in ["healthcare", "medical", "hipaa"]):
            if "hipaa" in name:
                combined += 0.30
            if "medical terminology" in name:
                combined += 0.25

        # TECH
        if "aws" in query_lower:
            if "amazon web services" in name:
                combined += 0.35
        if "docker" in query_lower:
            if "docker" in name:
                combined += 0.35
        if "java" in query_lower:
            if "core java" in name:
                combined += 0.25

        # NUMERICAL / COGNITIVE
        if "numerical" in query_lower:
            if "numerical" in combined_text:
                combined += 0.10
        if "cognitive" in query_lower:
            if "ability" in combined_text:
                combined += 0.10
            if "reasoning" in combined_text:
                combined += 0.10

        # DSI / SVAR explicit
        if "dsi" in query_lower:
            if "dependability and safety instrument" in name:
                combined += 0.40
        if "svar" in query_lower:
            if "svar spoken english" in name:
                combined += 0.40

        combined_scores[idx] = combined

    # ---- SORT ----
    sorted_indices = sorted(
        combined_scores,
        key=lambda i: combined_scores[i],
        reverse=True
    )

    # ---- DIVERSITY FILTERING ----
    results    = []
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