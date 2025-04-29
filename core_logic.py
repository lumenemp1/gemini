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

# ----------------------------------------------------------------------
# Configuration – edit paths / model names/dialect here
# ----------------------------------------------------------------------
INDEX_PATH        = "../schema_index/faiss_index.bin"
META_PATH         = "../schema_index/table_metadata.json"
EMBED_MODEL_NAME  = "BAAI/bge-small-en"
TOP_K             = 3
DB_URI            = "mysql+pymysql://root:admin@localhost/chatbot"
DIALECT           = "MySQL"  # Injected dialect for prompt

# Gemini API settings (hard-coded)
GEMINI_API_KEY    = ""
GEMINI_MODEL      = "gemini-2.0-flash"
GEMINI_URL        = (
    f"https://generativelanguage.googleapis.com/v1beta/models/"
    f"{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
)

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
def generate_sql_with_gemini(full_prompt: str) -> str:
    """
    Sends the prompt to Gemini API and returns the generated SQL string.
    Handles various response formats, including nested 'parts'.
    """
    payload = {"contents": [{"parts": [{"text": full_prompt}]}]}
    headers = {"Content-Type": "application/json"}
    resp = requests.post(GEMINI_URL, headers=headers, json=payload)
    resp.raise_for_status()
    data = resp.json()

    # Top-level 'candidates' list
    if "candidates" in data and isinstance(data["candidates"], list):
        candidate = data["candidates"][0]
        # If content is a dict with 'parts'
        if isinstance(candidate.get("content"), dict) and "parts" in candidate["content"]:
            parts = candidate["content"]["parts"]
            if parts and isinstance(parts[0].get("text"), str):
                return parts[0]["text"]
        # If content is a string
        if isinstance(candidate.get("content"), str):
            return candidate["content"]
        # Other possible fields
        for key in ("text", "output"):  
            if isinstance(candidate.get(key), str):
                return candidate[key]
    # Fallback: direct top-level
    for key in ("content", "text"):  
        if isinstance(data.get(key), str):
            return data[key]

    # If none matched, raise with full JSON for debugging
    raise ValueError(f"Unexpected Gemini response format: {json.dumps(data)}")

# ----------------------------------------------------------------------
# Public Init and Inference
# ----------------------------------------------------------------------
def init_models():
    global _faiss_index, _metadata, _rev_fk_map, _embed_model
    print("🔧 Initializing FAISS and Embeddings…")
    _faiss_index, _metadata = load_faiss_and_metadata(INDEX_PATH, META_PATH)
    _rev_fk_map = build_reverse_fk_map(_metadata)
    _embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    print("✅ FAISS and Embeddings ready!")






def generate_chart_suggestions(question: str, schema: str, sql_query: str) -> Dict[str, Any]:
    prompt = f"""
You are a data visualization assistant.

Given the following user question, SQL query, and database schema, suggest all suitable Plotly Express chart types to visualize the query results. For each chart, provide a JSON object with a "chart_type" matching a Plotly Express function name (e.g., "bar", "line", "scatter", "histogram", "box", "pie", etc.), and any relevant keyword arguments like "x", "y", "z", "color", "size", "dimensions", "names", "values", "path", etc.

Supported Plotly Express chart types: bar, line, scatter, histogram, box, pie, density_heatmap, area, funnel, treemap, sunburst, violin, scatter_3d, surface, parallel_coordinates, parallel_categories, choropleth, choropleth_mapbox.

### User Question:
{question}

### SQL Query:
{sql_query}

### Schema:
{schema}

### Response Format:
{{
  "charts": [
    {{ "chart_type": "bar", "x": "product_name", "y": "total_sales" }},
    {{ "chart_type": "pie", "names": "product_name", "values": "total_sales" }}
  ]
}}
"""
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    headers = {"Content-Type": "application/json"}
    response = requests.post(GEMINI_URL, headers=headers, json=payload)
    response.raise_for_status()
    data = response.json()

    charts_json = None
    if "candidates" in data and data["candidates"]:
        candidate = data["candidates"][0]
        content = candidate.get("content")
        if isinstance(content, dict) and "parts" in content:
            charts_json = content["parts"][0].get("text")
        elif isinstance(content, str):
            charts_json = content
    if charts_json:
        try:
            return json.loads(charts_json)
        except json.JSONDecodeError:
            cleaned = charts_json.strip().strip("```json").strip("```")
            return json.loads(cleaned)
    return {"charts": []}

def plot_charts_from_config(df: pd.DataFrame, chart_config: Dict[str, Any]) -> List[Any]:
    figs: List[Any] = []
    for chart in chart_config.get("charts", []):
        chart_type = chart.get("chart_type", "").lower().replace(" ", "_")
        plot_fn = getattr(px, chart_type, None)
        kwargs = {k: v for k, v in chart.items() if k != "chart_type"}

        if callable(plot_fn):
            try:
                fig = plot_fn(df, **kwargs)
            except Exception:
                fig = fallback_table(df)
        else:
            fig = fallback_table(df)
        figs.append(fig)
    return figs

def fallback_table(df: pd.DataFrame) -> go.Figure:
    return go.Figure(data=[go.Table(
        header=dict(values=list(df.columns)),
        cells=dict(values=[df[col] for col in df.columns])
    )])








def process_question(question: str) -> dict:
    try:
        # 1) semantic search for relevant tables
        idxs        = semantic_search(question, _embed_model, _faiss_index, TOP_K)
        tables      = expand_with_related(idxs, _metadata, _rev_fk_map)
        schema_text = build_schema_snippet(tables, _metadata)

        # 2) format prompt
        prompt      = PROMPT_TEMPLATE.format(
            question=question,
            schema=schema_text,
            dialect=DIALECT
        )

        # 3) get SQL from Gemini
        gen_output = generate_sql_with_gemini(prompt)
        if not isinstance(gen_output, str):
            raise ValueError(f"Gemini returned non-text: {type(gen_output)}")

        final_sql = gen_output.strip().lstrip("```sql").rstrip("```").strip()

        # 4) execute SQL
        engine = create_engine(DB_URI)
        with engine.connect() as conn:
            rows = [dict(r._mapping) for r in conn.execute(text(final_sql)).fetchall()]
        df = pd.DataFrame(rows)

        # 5) chart generation
        chart_config = generate_chart_suggestions(question, schema_text, final_sql)
        figures = plot_charts_from_config(df, chart_config)
        chart_jsons = [fig.to_json() for fig in figures]

        return {
            "sql": final_sql,
            "results": rows,
            "charts": chart_jsons
        }

    except Exception as e:
        return {"sql": None, "results": [], "charts": [], "error": str(e)}
