import json
import pickle
import re
import numpy as np
import faiss

from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

# =========================================================
# LOAD CATALOG
# =========================================================

with open("shl_product_catalog.json", "r", encoding="utf-8") as f:
    catalog = json.load(f)

print(f"Loaded {len(catalog)} assessments")

# =========================================================
# TEST TYPE MAPPING
# =========================================================

KEY_TO_TYPE = {
    "Knowledge & Skills": "K",
    "Personality & Behavior": "P",
    "Ability & Aptitude": "A",
    "Biodata & Situational Judgment": "B",
    "Simulations": "S",
    "Competencies": "C",
    "Development & 360": "D",
    "Assessment Exercises": "E",
}

# =========================================================
# TEXT CLEANING
# =========================================================

def clean_text(text):
    if not text:
        return ""

    text = str(text)

    # remove weird whitespace
    text = re.sub(r"\s+", " ", text)

    # remove weird characters
    text = text.replace("\n", " ").replace("\r", " ")

    return text.strip()

# =========================================================
# TEST TYPE EXTRACTION
# =========================================================

def get_test_type(keys):
    types = []

    for key in keys:
        if key in KEY_TO_TYPE:
            mapped = KEY_TO_TYPE[key]

            if mapped not in types:
                types.append(mapped)

    return ",".join(types) if types else "K"

# =========================================================
# ENRICH SEARCH TEXT
# =========================================================

def make_search_text(item):

    name = clean_text(item.get("name", ""))

    description = clean_text(item.get("description", ""))

    job_levels = ", ".join(item.get("job_levels", []))

    languages = ", ".join(item.get("languages", []))

    keys = ", ".join(item.get("keys", []))

    duration = clean_text(item.get("duration", ""))

    remote = clean_text(item.get("remote", ""))

    adaptive = clean_text(item.get("adaptive", ""))

    # IMPORTANT:
    # repeat important fields slightly for retrieval boosting

    text = f"""
    Assessment Name: {name}

    {name} assessment.

    Description:
    {description}

    Assessment Categories:
    {keys}

    Suitable Job Levels:
    {job_levels}

    Supported Languages:
    {languages}

    Duration:
    {duration}

    Remote Testing:
    {remote}

    Adaptive Testing:
    {adaptive}

    Skills and Assessment Focus:
    {description}

    """

    return clean_text(text)

# =========================================================
# CLEAN + ENRICH CATALOG
# =========================================================

processed_catalog = []

seen_urls = set()

for item in catalog:

    url = item.get("link", "").strip()

    # remove duplicates
    if not url or url in seen_urls:
        continue

    seen_urls.add(url)

    item["name"] = clean_text(item.get("name", ""))

    item["description"] = clean_text(item.get("description", ""))

    item["duration"] = clean_text(item.get("duration", ""))

    item["test_type"] = get_test_type(item.get("keys", []))

    item["search_text"] = make_search_text(item)

    processed_catalog.append(item)

catalog = processed_catalog

print(f"Final cleaned catalog size: {len(catalog)}")

# =========================================================
# LOAD EMBEDDING MODEL
# =========================================================

print("Loading embedding model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

# =========================================================
# BUILD EMBEDDINGS
# =========================================================

print("Building embeddings...")

texts = [item["search_text"] for item in catalog]

embeddings = model.encode(
    texts,
    show_progress_bar=True,
    batch_size=32,
    convert_to_numpy=True,
    normalize_embeddings=True
)

embeddings = np.array(embeddings, dtype="float32")

# =========================================================
# BUILD FAISS INDEX
# =========================================================

print("Building FAISS index...")

dimension = embeddings.shape[1]

index = faiss.IndexFlatIP(dimension)

index.add(embeddings)

# =========================================================
# BUILD BM25
# =========================================================

print("Building BM25 index...")

tokenized_corpus = [
    text.lower().split()
    for text in texts
]

bm25 = BM25Okapi(tokenized_corpus)

# =========================================================
# SAVE EVERYTHING
# =========================================================

print("Saving files...")

faiss.write_index(index, "catalog.index")

with open("catalog_data.pkl", "wb") as f:
    pickle.dump(
        {
            "catalog": catalog,
            "bm25": bm25,
        },
        f
    )

print("====================================")
print("DONE! Retrieval index built successfully.")
print(f"Total assessments indexed: {len(catalog)}")
print("Saved:")
print(" - catalog.index")
print(" - catalog_data.pkl")
print("====================================")