import os
import json
import re
import numpy as np
import faiss
import requests
from sentence_transformers import SentenceTransformer
from sqlalchemy import create_engine, text
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Any
import os
from dotenv import load_dotenv

# ----------------------------------------------------------------------
# Configuration – edit paths / model names/dialect here
# ----------------------------------------------------------------------
EMBED_MODEL_NAME = "BAAI/bge-small-en"
TOP_K = 3
DIALECT = "MySQL"

load_dotenv()

# Retrieve the API key
api_key = os.getenv("GEMINI_API_KEY")

# Gemini API settings (hard-coded)

GEMINI_MODEL = "gemini-2.0-flash"
GEMINI_URL = (
    f"https://generativelanguage.googleapis.com/v1beta/models/"
    f"{GEMINI_MODEL}:generateContent?key={api_key}"
)

# ----------------------------------------------------------------------
# DB registry
# ----------------------------------------------------------------------
db_configs = {
    "chatbot": {
        "db_uri": "mysql+pymysql://root:admin@localhost/chatbot",
        "index_path": "schema_index/faiss_index.bin",
        "meta_path": "schema_index/table_metadata.json"
    },
    "pipeline": {
        "db_uri": "mysql+pymysql://root:admin@localhost/pipeline",
        "index_path": "schema_index/pipeline_faiss_index.bin",
        "meta_path": "schema_index/pipeline_metadata.json"
    },
    "swift": {
        "db_uri": "mysql+pymysql://root:admin@localhost/swift_orders",
        "index_path": "schema_index/swift_faiss_index.bin",
        "meta_path": "schema_index/swift_metadata.json"
    }
}

# This will store preloaded resources per DB
DB_RESOURCES: Dict[str, dict] = {}

# ----------------------------------------------------------------------
# Prompt Template with system instruction to limit output to SQL
# ----------------------------------------------------------------------
PROMPT_TEMPLATE = """
You are a database assistant. Only provide the SQL query without any explanations or extra text.

### Task
Generate a {dialect}-compatible SQL query to answer the following question:
{question}

### Database Schema
The query will run on a {dialect} database with the following schema:
{schema}

### Guidelines
1. Ensure the query is fully compatible with {dialect}.
2. Avoid unsupported syntax (e.g., `NULLS LAST`).
3. Handle `NULL` explicitly (e.g., `CASE WHEN`).

### SQL Query
```sql
"""

# ----------------------------------------------------------------------
# Load FAISS + metadata (unchanged)
# ----------------------------------------------------------------------

def load_faiss_and_metadata(index_path, meta_path):
    index = faiss.read_index(index_path)
    with open(meta_path, "r") as f:
        meta = json.load(f)
    return index, meta


def build_reverse_fk_map(metadata):
    rev = {m["table_name"]: set() for m in metadata.values()}
    fk_re = re.compile(r"REFERENCES\s+`?(\w+)`?", re.IGNORECASE)
    for m in metadata.values():
        for ref in fk_re.findall(m["create_stmt"]):
            if ref in rev:
                rev[ref].add(m["table_name"])
    return rev


def parse_forward_fks(ddl):
    return set(re.findall(r"REFERENCES\s+`?(\w+)`?", ddl, flags=re.IGNORECASE))


def semantic_search(q, embed_model, faiss_index, top_k):
    emb = embed_model.encode(q)
    D, I = faiss_index.search(np.array([emb], dtype="float32"), top_k)
    return I[0]


def expand_with_related(idxs, metadata, rev_map):
    tables = {metadata[str(i)]["table_name"] for i in idxs}
    extra = set()
    for i in idxs:
        m = metadata[str(i)]
        extra |= parse_forward_fks(m["create_stmt"])
        extra |= rev_map.get(m["table_name"], set())
    return tables | extra


def build_schema_snippet(tables, metadata):
    return "\n\n".join(
        m["create_stmt"] for m in metadata.values() if m["table_name"] in tables
    )

# ----------------------------------------------------------------------
# Gemini integration (enhanced parsing for nested 'parts')
# ----------------------------------------------------------------------

import time

def generate_sql_with_gemini(full_prompt: str, max_retries: int = 3, delay: float = 2.0) -> str:
    """
    Sends the prompt to Gemini API and returns the generated SQL string.
    Retries on 503 Service Unavailable errors.
    """
    payload = {"contents": [{"parts": [{"text": full_prompt}]}]}
    headers = {"Content-Type": "application/json"}

    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(GEMINI_URL, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()

            # Top-level 'candidates' list
            if "candidates" in data and isinstance(data["candidates"], list):
                candidate = data["candidates"][0]
                content = candidate.get("content")

                if isinstance(content, dict) and "parts" in content:
                    parts = content["parts"]
                    if parts and isinstance(parts[0].get("text"), str):
                        return parts[0]["text"]
                elif isinstance(content, str):
                    return content
                for key in ("text", "output"):
                    if isinstance(candidate.get(key), str):
                        return candidate[key]

            for key in ("content", "text"):
                if isinstance(data.get(key), str):
                    return data[key]

            raise ValueError(f"Unexpected Gemini response format: {json.dumps(data)}")

        except requests.exceptions.HTTPError as http_err:
            if resp.status_code == 503:
                print(f"[Retry {attempt}] Gemini API returned 503. Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                raise http_err
        except Exception as other_err:
            print(f"[Retry {attempt}] Error: {other_err}. Retrying in {delay} seconds...")
            time.sleep(delay)

    raise RuntimeError("Gemini API failed after multiple retries.")

# ----------------------------------------------------------------------
# Public Init and Inference
# ----------------------------------------------------------------------
def init_all_db_resources():
    print("🔧 Loading all DB resources...")
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    for db_key, cfg in db_configs.items():
        print(f"⏳ Loading DB: {db_key}")
        index, metadata = load_faiss_and_metadata(cfg["index_path"], cfg["meta_path"])
        rev_fk = build_reverse_fk_map(metadata)
        DB_RESOURCES[db_key] = {
            "embed_model": embed_model,
            "faiss_index": index,
            "metadata": metadata,
            "rev_fk_map": rev_fk,
            "db_uri": cfg["db_uri"]
        }
    print("✅ All DB resources loaded.")




def retry_sql_with_error_context(
    question: str,
    schema: str,
    previous_sql: str,
    error_msg: str,
    dialect: str = "MySQL",
    max_retries: int = 2
) -> str:
    """
    Retry SQL generation with Gemini by feeding the DB error back into the prompt.
    Tries up to `max_retries` times and returns corrected SQL string.
    """
    for attempt in range(max_retries):
        prompt = f"""
You are a SQL assistant helping correct errors.

The following SQL query caused an error. Your task is to revise the query to fix the error **without** adding any explanations. Only return the corrected {dialect}-compatible SQL query.

### User Question
{question}

### Database Schema
{schema}

### Previous SQL Query
```sql
{previous_sql}
{error_msg}
"""

        try:
            corrected_sql = generate_sql_with_gemini(prompt)
            return corrected_sql.strip().lstrip("```sql").rstrip("```").strip()
        except Exception as retry_err:
            print(f"[Retry {attempt+1}] Gemini correction failed: {retry_err}")
            continue

    raise RuntimeError("Maximum retries exceeded while attempting to fix the SQL query.")



def process_question(question: str, selected_dbs: List[str]) -> List[dict]:
    results = []

    for db_key in selected_dbs:
        try:
            resource = DB_RESOURCES[db_key]
            embed_model = resource["embed_model"]
            faiss_index = resource["faiss_index"]
            metadata = resource["metadata"]
            rev_fk_map = resource["rev_fk_map"]
            db_uri = resource["db_uri"]

            # 1. Semantic search
            idxs = semantic_search(question, embed_model, faiss_index, TOP_K)
            tables = expand_with_related(idxs, metadata, rev_fk_map)
            schema_text = build_schema_snippet(tables, metadata)

            # 2. Prompt creation
            prompt = PROMPT_TEMPLATE.format(
                question=question,
                schema=schema_text,
                dialect=DIALECT
            )

            # 3. Gemini SQL generation
            gen_output = generate_sql_with_gemini(prompt)
            if not isinstance(gen_output, str):
                raise ValueError(f"Gemini returned non-text: {type(gen_output)}")

            final_sql = gen_output.strip().lstrip("```sql").rstrip("```").strip()

            # 4. SQL execution (with retry on error)
            engine = create_engine(db_uri)
            try:
                with engine.connect() as conn:
                    rows = [dict(r._mapping) for r in conn.execute(text(final_sql)).fetchall()]
            except Exception as db_err:
                print(f"[{db_key}] SQL error, retrying with error context: {db_err}")
                corrected_sql = retry_sql_with_error_context(
                    question, schema_text, final_sql, str(db_err), DIALECT
                )
                with engine.connect() as conn:
                    rows = [dict(r._mapping) for r in conn.execute(text(corrected_sql)).fetchall()]
                final_sql = corrected_sql

            results.append({
                "db": db_key,
                "sql": final_sql,
                "results": rows
            })

        except Exception as e:
            results.append({
                "db": db_key,
                "sql": None,
                "results": [],
                "error": str(e)
            })

    return results
