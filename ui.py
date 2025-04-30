import chainlit as ch
import aiohttp
import datetime
import asyncio
import json
import plotly.io as pio

# Backend API endpoint
url = "http://127.0.0.1:5000/api/query"

@ch.on_chat_start
async def start_chat():
    now = datetime.datetime.now().hour
    if now < 12:
        greeting = "Good Morning! How can I assist you today?"
    elif now < 18:
        greeting = "Good Afternoon! How can I assist you today?"
    else:
        greeting = "Good Evening! How can I assist you today?"
    await ch.Message(content=greeting).send()

@ch.on_message
async def main(message):
    await ch.Message(content="🔄 Processing your query… This may take a few seconds.").send()

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                json={"question": message.content},
                timeout=aiohttp.ClientTimeout(total=600)
            ) as response:

                if response.status != 200:
                    error = await response.text()
                    await ch.Message(content=f"❌ Server error {response.status}: {error}").send()
                    return

                data = await response.json()

                # SQL query
                sql = data.get('sql', '')
                await ch.Message(content=f"### 🧠 Generated SQL Query\n```sql\n{sql}\n```").send()

                # Table results
                results = data.get('results', [])
                if results:
                    headers = list(results[0].keys())
                    md  = "### 📋 Query Results\n\n| " + " | ".join(headers) + " |\n"
                    md += "| " + " | ".join(["---"] * len(headers)) + " |\n"
                    for row in results:
                        vals = [str(row.get(h, "")).replace("|", "\\|") for h in headers]
                        md += "| " + " | ".join(vals) + " |\n"
                    # send table and keep its Message object
                    table_msg = await ch.Message(content=md).send()
                else:
                    await ch.Message(content="⚠️ No results found for your query.").send()
                    table_msg = None

                # Chart rendering
                charts = data.get('charts', [])
                if charts:
                    await ch.Message(content="📊 Generating charts based on query…").send()
                    for idx, chart_json in enumerate(charts, start=1):
                        try:
                            fig = pio.from_json(chart_json)
                            # attach chart to the table message so it appears below
                            if table_msg:
                                await ch.Plotly(name=f"Chart {idx}", figure=fig).send(table_msg.id)
                            else:
                                # fallback if no table
                                await ch.Plotly(name=f"Chart {idx}", figure=fig).send()
                        except Exception as err:
                            await ch.Message(content=f"⚠️ Could not render Chart {idx}: {err}").send()
                else:
                    await ch.Message(content="ℹ️ No charts were suggested for this query.").send()

    except asyncio.TimeoutError:
        await ch.Message(content="⏱️ The model timed out. Try a simpler query.").send()
    except aiohttp.ClientConnectorError as conn_err:
        await ch.Message(content=f"🔌 Could not connect to the backend: {conn_err}").send()
    except Exception as e:
        await ch.Message(content=f"🔥 Unexpected error: {type(e).__name__} - {e}").send()
