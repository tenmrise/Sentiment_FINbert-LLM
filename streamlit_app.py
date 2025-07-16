import streamlit as st
import os
import json
import datetime as dt
from typing import Dict, List, Any
import numpy as np

from dotenv import load_dotenv
import pandas as pd
import yfinance as yf
import plotly.express as px

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEndpoint
from langchain_community.tools.tavily_search import TavilySearchResults

# ==============================================================================
# 1. CORE APPLICATION LOGIC
# ==============================================================================

# --------  CONFIG  --------
CONFIG = {
    "max_articles_per_ticker": 5,
    "extractor_model": "gpt-4o",
    "analyst_model": "gemini-2.5-pro",
    "sentiment_model_repo_id": "ProsusAI/finbert",
}

# --------  API KEYS & MODELS  --------
@st.cache_resource
def load_models_and_keys():
    load_dotenv()
    keys = {"TAVILY_API_KEY": os.getenv("TAVILY_API_KEY"), "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),"GOOGLE_API_KEY": os.getenv("GOOGLE_API_KEY"),"HUGGINGFACEHUB_API_TOKEN": os.getenv("HUGGINGFACEHUB_API_TOKEN")}
    if not all(keys.values()):
        st.error("API Key Missing! Ensure TAVILY, OPENAI, GOOGLE, and HUGGINGFACEHUB keys are in secrets.")
        return None, None
    models = {
        "search_tool": TavilySearchResults(max_results=CONFIG["max_articles_per_ticker"]),
        "fact_extractor_llm": ChatOpenAI(model=CONFIG["extractor_model"], temperature=0.0, api_key=keys["OPENAI_API_KEY"]),
        "analyst_llm": ChatGoogleGenerativeAI(model=CONFIG["analyst_model"], temperature=0.1, model_kwargs={"response_mime_type": "application/json"}, api_key=keys["GOOGLE_API_KEY"]),
        # --- FIX 1: Explicitly set the correct task for the model ---
        "sentiment_analyzer": HuggingFaceEndpoint(
            repo_id=CONFIG["sentiment_model_repo_id"],
            task="text-classification",
            huggingfacehub_api_token=keys["HUGGINGFACEHUB_API_TOKEN"],
        )
    }
    return keys, models

# -------- PROMPTS (With formatting fix) --------
fact_extraction_prompt = ChatPromptTemplate.from_messages([("system", "You are a data extraction engine. From the provided news article text, extract the following information. Do not interpret, analyze, or add any information not present in the text. Your output must be a JSON object with the keys 'key_figures', 'core_event', and 'outlook'. If a key is not mentioned in the text, its value should be 'Not mentioned'."),("human", "Article Text:\n```{article_text}```")])
# --- FIX 3: Cleaned up the prompt string to have valid variable names ---
aggregate_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        (
            "You are a quantitative investment strategist. Your task is to synthesize three independent streams of pre-computed data: quantitative sentiment, extracted factual catalysts, and market price action. Your goal is to identify **divergence** or **convergence** between the news narrative and the stock's performance.\n\n"
            "**Analytical Framework:**\n"
            "1.  **Review Sentiment Profile:** Is the statistical sentiment profile Positive, Negative, or Contentious (high standard deviation)?\n"
            "2.  **Review Factual Catalysts:** Do the extracted key facts represent clear positive or negative events?\n"
            "3.  **Review Price Action:** Did the stock significantly outperform, underperform, or track the market?\n"
            "4.  **Formulate Thesis:** Synthesize the three data streams. Is there a clear DIVERGENCE? (e.g., 'Despite a negative sentiment profile and no clear positive catalysts, the stock remained resilient, suggesting the market has already priced in known risks.') Or is there a CONVERGENCE? (e.g., 'Strong positive sentiment, driven by the new product launch, is confirmed by the stock's significant outperformance.')\n\n"
            "**Output Schema (Strict JSON):**\n"
            "```json\n"
            "{\n"
            "  \"ticker\": \"<The stock ticker>\",\n"
            "  \"investment_thesis\": \"<Your concise thesis based on the divergence/convergence analysis>\",\n"
            "  \"final_score\": <Integer from 1 to 10>,\n"
            "  \"score_justification\": \"<1-sentence justification for your score, citing the thesis.>\"\n"
            "}\n"
            "```"
        )
    ),
    (
        "human",
        "Input Data for Ticker: {ticker}\n\n"
        "**Quantitative Sentiment Profile:**\n```{sent_analysis}```\n\n"
        "**Extracted Factual Catalysts:**\n```{key_facts}```\n\n"
        "**Quantitative Price Action:**\n```{fin_analysis}```"
    )
])

# -------- HELPER FUNCTIONS (With bug fixes) --------
def fetch_news(ticker, start_date, end_date, search_tool):
    try:
        query = f"stock market news for {ticker} between {start_date:%Y-%m-%d} and {end_date:%Y-%m-%d}"
        articles = search_tool.invoke(query)
        renamed_articles = [{"title": art.get("title"), "content": art.get("content"), "link": art.get("url")} for art in articles]
        return renamed_articles
    except Exception as e:
        st.error(f"Error fetching news for {ticker}: {e}")
        return []

def extract_key_facts(articles, fact_extraction_chain):
    if not articles: return []
    try:
        batch_inputs = [{"article_text": (art.get("content") or art.get("title", ""))} for art in articles]
        batch_results = fact_extraction_chain.batch(batch_inputs, {"max_concurrency": 5})
        for i, article in enumerate(articles):
            if i < len(batch_results): article["key_facts"] = batch_results[i]
        return articles
    except Exception as e:
        st.error(f"Error extracting facts: {e}")
        return articles

def score_sentiment(articles, sentiment_analyzer):
    if not articles: return [], {}
    try:
        texts_to_score = [(art.get("content") or art.get("title", "")) for art in articles]
        api_results = sentiment_analyzer.batch(texts_to_score)
        scores = []
        for i, article in enumerate(articles):
            if i < len(api_results):
                pmap = {d["label"].lower(): d["score"] for d in api_results[i]}
                score = round(pmap.get("positive", 0) - pmap.get("negative", 0), 3)
                article["sentiment_score"] = score; scores.append(score)
        if scores:
            sdf = pd.DataFrame(scores, columns=["score"])
            sentiment_analysis = {
                "average_score": float(sdf["score"].mean()),
                "std_dev_sentiment": float(sdf["score"].std()) if len(scores) > 1 else 0.0,
                "num_articles": len(scores),
                "num_positive": int(sdf[sdf["score"] > 0.1].count().iloc[0]),
                "num_negative": int(sdf[sdf["score"] < -0.1].count().iloc[0]),
                "num_neutral": int(sdf[(sdf["score"] >= -0.1) & (sdf["score"] <= 0.1)].count().iloc[0])
            }
            return articles, sentiment_analysis
        return articles, {}
    except Exception as e:
        st.error(f"Error scoring sentiment: {e}")
        return articles, {}

def fetch_and_analyse_finance(ticker, start_date, end_date):
    try:
        end_date_adj = end_date + dt.timedelta(days=1)
        df = yf.download(ticker, start=start_date, end=end_date_adj, progress=False, ignore_tz=True)
        if df.empty or len(df) < 2: raise ValueError("Not enough historical data found.")
        n_day_return = (df['Close'].iloc[-1] / df['Close'].iloc[0]) - 1
        largest_move = df['Close'].pct_change().abs().max()
        # --- FIX 2: Use pd.notna() to correctly check for a valid value ---
        finance_analysis = {
            "period_return_pct": float(round(n_day_return * 100, 2)),
            "largest_daily_move_pct": float(round(largest_move * 100, 2)) if pd.notna(largest_move) else 0.0
        }
        return df, finance_analysis
    except Exception as e:
        st.error(f"Error analyzing finance for {ticker}: {e}")
        return pd.DataFrame(), {}

# ==============================================================================
# 2. STREAMLIT DASHBOARD UI
# ==============================================================================
st.set_page_config(layout="wide", page_title="Financial Sentiment Dashboard")
st.title("📈 Financial News & Sentiment Analysis")
st.markdown("A dashboard for quantitative synthesis of news sentiment and price action, powered by LLM agents.")

api_keys, models = load_models_and_keys()
if not models: st.stop()

@st.cache_data(show_spinner=False)
def run_analysis_for_one_ticker(_models, _ticker, _start_date, _end_date):
    fact_extraction_chain = fact_extraction_prompt | _models['fact_extractor_llm'] | JsonOutputParser()
    aggregate_chain = aggregate_prompt | _models['analyst_llm'] | JsonOutputParser()
    articles = fetch_news(_ticker, _start_date, _end_date, _models['search_tool'])
    articles_with_facts = extract_key_facts(articles, fact_extraction_chain)
    articles_with_sentiment, sentiment_stats = score_sentiment(articles_with_facts, _models['sentiment_analyzer'])
    price_df, finance_stats = fetch_and_analyse_finance(_ticker, _start_date, _end_date)
    key_facts_for_prompt = [a.get("key_facts", {}) for a in articles_with_sentiment]
    try:
        final_report = aggregate_chain.invoke({
            "ticker": _ticker,
            "sent_analysis": json.dumps(sentiment_stats),
            "key_facts": json.dumps(key_facts_for_prompt),
            "fin_analysis": json.dumps(finance_stats)
        })
    except Exception as e:
        st.error(f"Error generating final report for {_ticker}: {e}")
        final_report = None
    return {"ticker": _ticker, "report": final_report, "news_data": articles_with_sentiment, "finance_analysis": finance_stats, "prices_raw": price_df}

st.sidebar.header("Analysis Configuration")
if 'available_tickers' not in st.session_state: st.session_state.available_tickers = ['NVDA', 'GOOGL', 'MSFT', 'AAPL']
new_ticker = st.sidebar.text_input("Add Ticker Symbol", placeholder="e.g., CRM").strip().upper()
if st.sidebar.button("Add Ticker"):
    if new_ticker and new_ticker not in st.session_state.available_tickers: st.session_state.available_tickers.append(new_ticker)
    elif not new_ticker: st.sidebar.warning("Please enter a ticker symbol.")
    else: st.sidebar.warning(f"{new_ticker} is already in the list.")
selected_tickers = st.sidebar.multiselect("Select Stock Tickers for Analysis", options=st.session_state.available_tickers, default=['NVDA', 'MSFT'])
today = dt.date.today()
start_date = st.sidebar.date_input("Start Date", value=today - dt.timedelta(days=7))
end_date = st.sidebar.date_input("End Date", value=today)

if st.sidebar.button("🚀 Run Analysis", type="primary"):
    if not selected_tickers: st.warning("Please select at least one ticker.")
    elif start_date >= end_date: st.warning("Start Date must be before End Date.")
    else:
        st.session_state.all_results = []; results = []; status_container = st.empty(); progress_bar = st.progress(0)
        for i, ticker in enumerate(selected_tickers):
            status_container.info(f"▶️ Now analyzing: **{ticker}** ({i+1} of {len(selected_tickers)})...")
            result = run_analysis_for_one_ticker(models, ticker, start_date, end_date)
            results.append(result); progress_bar.progress((i + 1) / len(selected_tickers))
        status_container.success("✅ Analysis Complete!"); st.session_state.all_results = results

if 'all_results' in st.session_state and st.session_state.all_results:
    results = st.session_state.all_results
    st.subheader("Executive Summary Report")
    summary_data = []
    successful_results = []
    for res in results:
        if not res or not res.get('report'): 
            summary_data.append({"Ticker": res.get('ticker', 'N/A'), "Score": "N/A", "Thesis": "ERROR: Analysis failed to complete."})
        else: 
            report = res['report']
            summary_data.append({"Ticker": report.get('ticker'), "Score": report.get('final_score'), "Thesis": report.get('investment_thesis')})
            successful_results.append(res)
    summary_df = pd.DataFrame(summary_data); st.dataframe(summary_df, use_container_width=True, hide_index=True)
    if successful_results:
        st.subheader("Detailed Analysis by Ticker")
        tab_titles = [res.get("ticker") for res in successful_results]
        tabs = st.tabs(tab_titles)
        for i, res in enumerate(successful_results):
            with tabs[i]:
                st.subheader(f"Data Synthesis for {res['ticker']}")
                report = res['report']
                st.info(f"**Investment Thesis:** {report.get('investment_thesis')}")
                st.write(f"**Justification:** {report.get('score_justification')}")
                st.subheader("Raw Data & Extracted Facts")
                col1, col2 = st.columns(2)
                col1.metric(label=f"Return ({start_date} to {end_date})", value=f"{res['finance_analysis'].get('period_return_pct', 0):.2f}%")
                col2.metric(label="Largest Single-Day Move", value=f"{res['finance_analysis'].get('largest_daily_move_pct', 0):.2f}%")
                prices_df = res.get("prices_raw")
                if prices_df is not None and not prices_df.empty:
                    fig = px.line(prices_df, y="Close", title=f"{res['ticker']} Stock Price")
                    st.plotly_chart(fig, use_container_width=True)
                news_data = res.get("news_data", []) 
                if not news_data: st.info(f"No news articles found.")
                for item in news_data:
                    score = item.get('sentiment_score', 0); color = "green" if score > 0 else "red" if score < 0 else "blue"
                    with st.expander(f"**{item.get('title', 'Article')}** | Score: :{color}[{score:.3f}]"):
                        st.write("**Key Facts Extracted by LLM:**"); st.json(item.get('key_facts', 'No facts extracted.'))
                        st.write("**Original Content:**"); st.write(item.get('content'))
                        st.link_button("Go to Article", item.get("link", "#"))
else:
    st.info("Configure your analysis in the sidebar and click 'Run Analysis' to begin.")
