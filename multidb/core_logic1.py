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


import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Any


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
    "eon": {
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




def generate_chart_suggestions(
    question: str, 
    all_results: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Use Gemini to suggest a standardized chart config based on actual result columns (not SQL).
    """

    # Compose column info from result sets
    description_section = ""
    for result in all_results:
        db = result["db"]
        rows = result["results"]
        if not rows:
            continue
        sample_row = rows[0]
        description_section += f"-- {db} columns: {list(sample_row.keys())}\n"

    prompt = f"""
You are a data visualization assistant helping unify result sets from multiple databases.

The user question is:
{question}

Below are result columns returned from each database (not SQL):

{description_section}

Your tasks:
1. Understand the semantic meaning of each column in each DB.
2. Suggest a unified schema for charting (e.g., x_axis: product, y_axis: quantity).
3. Return a JSON config mapping actual column names to a standard schema.
4. Suggest a chart type for comparison.

### Desired Output JSON format:
{{
  "chart_type": "bar",
  "x_axis": {{
    "standard": "product_name",
    "eon": "product_name",
    "swift": "name"
  }},
  "y_axis": {{
    "standard": "total_orders",
    "eon": "order_qty",
    "swift": "stock_quantity"
  }},
  "group_by": "database"
}}
"""

    # Send to Gemini
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    headers = {"Content-Type": "application/json"}
    response = requests.post(GEMINI_URL, headers=headers, json=payload)
    response.raise_for_status()
    data = response.json()

    # Extract model response
    charts_json = None
    try:
        if "candidates" in data and data["candidates"]:
            candidate = data["candidates"][0]
            content = candidate.get("content")
            if isinstance(content, dict) and "parts" in content:
                charts_json = content["parts"][0].get("text", "")
            elif isinstance(content, str):
                charts_json = content
    except Exception as e:
        print("❌ Unexpected Gemini response structure.")
        print(json.dumps(data, indent=2))
        raise e

    # Parse safely using regex
    if charts_json:
        try:
            match = re.search(r"```json\s*(\{.*?\})\s*```", charts_json, re.DOTALL)
            if match:
                return json.loads(match.group(1))

            fallback_match = re.search(r"(\{.*?\})", charts_json, re.DOTALL)
            if fallback_match:
                return json.loads(fallback_match.group(1))

        except json.JSONDecodeError as e:
            print("❌ Failed to parse chart config. Raw Gemini output:")
            print(repr(charts_json))
            raise e

    print("⚠️ No valid chart configuration returned by Gemini.")
    return {}

# -----------------------------
# Plotting Helpers
# -----------------------------
def normalize_and_merge_results(all_results: List[Dict[str, Any]], chart_mapping: Dict[str, Any]) -> pd.DataFrame:
    frames = []
    x_std = chart_mapping["x_axis"].get("standard", "x")
    y_std = chart_mapping["y_axis"].get("standard", "y")

    for result in all_results:
        db_key = result["db"]
        rows = result["results"]
        if not rows:
            continue

        df = pd.DataFrame(rows)
        x_map = chart_mapping["x_axis"].get(db_key)
        y_map = chart_mapping["y_axis"].get(db_key)

        # Ensure x_axis exists — it's mandatory
        if not x_map or x_map not in df.columns:
            continue  # Skip this DB result if x-axis column is absent

        df = df.rename(columns={x_map: x_std})

        # Handle missing or absent y-axis gracefully
        if not y_map or y_map not in df.columns:
            df[y_std] = 1  # Default/fallback value to allow plotting
        else:
            df = df.rename(columns={y_map: y_std})

        df["database"] = db_key
        frames.append(df[[x_std, y_std, "database"]])

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

def fallback_table(df: pd.DataFrame) -> go.Figure:
    return go.Figure(data=[go.Table(
        header=dict(values=list(df.columns)),
        cells=dict(values=[df[col] for col in df.columns])
    )])

def plot_charts_from_config(df: pd.DataFrame, chart_config: Dict[str, Any]) -> List[Any]:
    figs: List[Any] = []

    # Default fallback
    charts = chart_config.get("charts")
    if not charts:
        base_chart = chart_config.get("chart_type", "bar")
        charts = [{"chart_type": t} for t in ["bar", "line", "scatter", "area"]]

    for chart in charts:
        chart_type = chart.get("chart_type", "bar").lower().replace(" ", "_")
        plot_fn = getattr(px, chart_type, None)

        kwargs = {
            "x": chart_config["x_axis"]["standard"],
            "y": chart_config["y_axis"]["standard"],
            "color": chart_config.get("group_by", "database")
        }

        if callable(plot_fn):
            try:
                fig = plot_fn(df, **kwargs)
            except Exception as e:
                print(f"⚠️ Plot failed for {chart_type}: {e}")
                fig = fallback_table(df)
        else:
            fig = fallback_table(df)

        figs.append(fig)

    return figs


#---------------------------
#main
#---------------------------

def process_question(question: str, selected_dbs: List[str]) -> Dict[str, Any]:
    all_results = []

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

            # 3. SQL generation
            gen_output = generate_sql_with_gemini(prompt)
            final_sql = gen_output.strip().lstrip("```sql").rstrip("```").strip()

            # 4. SQL execution
            engine = create_engine(db_uri)
            try:
                with engine.connect() as conn:
                    rows = [dict(r._mapping) for r in conn.execute(text(final_sql)).fetchall()]
            except Exception as db_err:
                print(f"[{db_key}] SQL error, retrying: {db_err}")
                corrected_sql = retry_sql_with_error_context(
                    question, schema_text, final_sql, str(db_err), DIALECT
                )
                with engine.connect() as conn:
                    rows = [dict(r._mapping) for r in conn.execute(text(corrected_sql)).fetchall()]
                final_sql = corrected_sql

            all_results.append({
                "db": db_key,
                "sql": final_sql,
                "results": rows
            })

        except Exception as e:
            all_results.append({
                "db": db_key,
                "sql": None,
                "results": [],
                "error": str(e)
            })

    sql_by_db = {res["db"]: res["sql"] for res in all_results if res.get("sql")}
    print(f"sql_by_db {sql_by_db}")
    print(f"all results{all_results}")
    # 5. Get chart suggestion from Gemini using SQL queries
    chart_config = generate_chart_suggestions(question,all_results)
                                        
    print(f"char_config{chart_config}")

    # 6. Normalize data for plotting
    merged_df = normalize_and_merge_results(all_results, chart_config)
    print(f"merged_df{merged_df}")

    # 7. Generate Plotly figures
    figures = plot_charts_from_config(merged_df, chart_config)
    figures_json=[fig.to_json() for fig in figures]

    return {
    "sql_queries": {r["db"]: r["sql"] for r in all_results if r.get("sql")},
    "db_results": {r["db"]: r["results"] for r in all_results},
    "charts": figures_json
    }
