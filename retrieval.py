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

model = SentenceTransformer(
    "all-MiniLM-L6-v2",
    device="cpu"
)

# =========================================================
# QUERY EXPANSION HINTS
# =========================================================

QUERY_HINTS = {

    # leadership / executive
    "leadership": [
        "executive",
        "manager",
        "leadership",
        "strategic",
        "director",
        "cxo",
    ],

    # personality
    "personality": [
        "behavior",
        "opq",
        "personality",
        "occupational personality questionnaire",
    ],

    # graduate hiring
    "graduate": [
        "entry-level",
        "campus",
        "graduate",
        "early career",
    ],

    # sales
    "sales": [
        "sales",
        "client",
        "customer",
        "commercial",
    ],

    # numerical
    "numerical": [
        "numerical",
        "quantitative",
        "reasoning",
        "arithmetic",
    ],

    # cognitive
    "cognitive": [
        "ability",
        "aptitude",
        "reasoning",
        "verbal",
        "inductive",
    ],

    # =====================================================
    # ABBREVIATION EXPANSIONS
    # =====================================================

    "gsa": [
        "global skills assessment",
        "global skills development report",
        "great 8 domains",
    ],

    "opq": [
        "occupational personality questionnaire",
        "opq32r",
        "opq32",
        "personality behavior",
    ],

    "sjt": [
        "situational judgment",
        "biodata",
        "scenarios",
        "work context decision",
    ],

    "mq": [
        "motivation questionnaire",
        "drives",
        "motivation",
    ],

    "verify": [
        "numerical reasoning",
        "verbal reasoning",
        "inductive reasoning",
        "verify",
    ]
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

    expanded_query = query + " " + " ".join(expanded_terms)

    return expanded_query.strip()

# =========================================================
# NAMED ASSESSMENT EXTRACTION
# =========================================================

def extract_named_assessments(query: str) -> list[str]:

    """
    Expands common abbreviations and named
    assessment references for comparison queries.
    """

    known_aliases = {

        "opq": "occupational personality questionnaire opq32r",

        "gsa": "global skills assessment global skills development report",

        "mq": "motivation questionnaire",

        "verify": "shl verify interactive",

        "sjt": "situational judgment graduate scenarios",
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

    """
    Hybrid retrieval:
    - semantic retrieval (FAISS)
    - BM25 keyword retrieval
    - abbreviation expansion
    - heuristic boosting
    - diversity filtering
    """

    # =====================================================
    # NAMED ENTITY EXPANSION
    # =====================================================

    named_assessments = extract_named_assessments(query)

    if named_assessments:
        query = query + " " + " ".join(named_assessments)

    # =====================================================
    # QUERY EXPANSION
    # =====================================================

    expanded_query = expand_query(query)

    query_lower = expanded_query.lower()

    # =====================================================
    # SEMANTIC SEARCH
    # =====================================================

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

    # =====================================================
    # BM25 SEARCH
    # =====================================================

    tokens = expanded_query.lower().split()

    bm25_scores_raw = bm25.get_scores(tokens)

    max_bm25 = max(bm25_scores_raw) if max(bm25_scores_raw) > 0 else 1

    bm25_scores = {
        i: float(score / max_bm25)
        for i, score in enumerate(bm25_scores_raw)
    }

    # =====================================================
    # CANDIDATE POOL
    # =====================================================

    candidate_indices = set(semantic_scores.keys()) | set(
        i for i, score in bm25_scores.items()
        if score > 0.10
    )

    combined_scores = {}

    # =====================================================
    # COMBINED SCORING
    # =====================================================

    for idx in candidate_indices:

        semantic = semantic_scores.get(idx, 0.0)

        bm25_score = bm25_scores.get(idx, 0.0)

        combined = (0.65 * semantic) + (0.35 * bm25_score)

        item = catalog[idx]

        name = item["name"].lower()

        description = item.get("description", "").lower()

        keys = " ".join(item.get("keys", [])).lower()

        combined_text = f"{name} {description} {keys}"

        # =================================================
        # LEADERSHIP BOOSTING
        # =================================================

        if "leadership" in query_lower:

            if "opq" in combined_text:
                combined += 0.12

            if "leadership" in combined_text:
                combined += 0.10

        # =================================================
        # PERSONALITY BOOSTING
        # =================================================

        if "personality" in query_lower or "opq" in query_lower:

            # prioritize actual OPQ32r instrument
            if (
                "opq32r" in name or
                "occupational personality questionnaire opq32r" in name
            ):
                combined += 0.25

            elif "opq" in name and "report" not in name:
                combined += 0.15

            elif "opq" in combined_text and "report" not in name:
                combined += 0.10

        # =================================================
        # GSA BOOSTING
        # =================================================

        if "gsa" in query_lower or "global skills" in query_lower:

            if "global skills assessment" in combined_text:
                combined += 0.25

            if "global skills development report" in combined_text:
                combined += 0.15

        # =================================================
        # GRADUATE BOOSTING
        # =================================================

        if "graduate" in query_lower:

            if "graduate scenarios" in combined_text:
                combined += 0.20

        # =================================================
        # NUMERICAL BOOSTING
        # =================================================

        if "numerical" in query_lower:

            if "numerical" in combined_text:
                combined += 0.10

        # =================================================
        # COGNITIVE BOOSTING
        # =================================================

        if "cognitive" in query_lower:

            if "ability" in combined_text:
                combined += 0.10

            if "reasoning" in combined_text:
                combined += 0.10

        combined_scores[idx] = combined

    # =====================================================
    # SORT RESULTS
    # =====================================================

    sorted_indices = sorted(
        combined_scores,
        key=lambda i: combined_scores[i],
        reverse=True
    )

    # =====================================================
    # DIVERSITY FILTERING
    # =====================================================

    results = []

    seen_names = set()

    for idx in sorted_indices:

        item = catalog[idx]

        base_name = (
            item["name"]
            .lower()
            .replace("1.0", "")
            .replace("2.0", "")
            .split("(")[0]
            .strip()
        )

        # avoid duplicate-ish variants
        if base_name in seen_names:
            continue

        seen_names.add(base_name)

        results.append({

            "name": item["name"],

            "url": item["link"],

            "test_type": item["test_type"],

            "description": item.get("description", ""),

            "job_levels": item.get("job_levels", []),

            "duration": item.get("duration", ""),

            "keys": item.get("keys", []),

            "score": round(combined_scores[idx], 4)
        })

        if len(results) >= top_k:
            break

    return results