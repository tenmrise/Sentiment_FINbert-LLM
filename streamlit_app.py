import streamlit as st
import os
import json
import datetime as dt
from typing import Dict, List, Any

from dotenv import load_dotenv
import pandas as pd
import yfinance as yf
import plotly.express as px

from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEndpoint
from langchain_community.utilities.google_serper import GoogleSerperAPIWrapper

# ==============================================================================
# 1. CORE PIPELINE LOGIC (This section is now stable and correct)
# ==============================================================================

# --------  CONFIG  --------
CONFIG = {
    "max_articles_per_ticker": 5,
    "summarizer_model": "gpt-4o",
    "analyst_model": "gemini-2.5-pro",
    "sentiment_model_repo_id": "ProsusAI/finbert",
}

# --------  API KEYS & MODELS  --------
@st.cache_resource
def load_models_and_keys():
    load_dotenv()
    keys = {
        "SERPER_API_KEY": os.getenv("SERPER_API_KEY"),
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "GOOGLE_API_KEY": os.getenv("GOOGLE_API_KEY"),
        "HUGGINGFACEHUB_API_TOKEN": os.getenv("HUGGINGFACEHUB_API_TOKEN")
    }
    if not all(keys.values()):
        st.error("API Key Missing! Ensure SERPER, OPENAI, GOOGLE, and HUGGINGFACEHUB keys are in your secrets.")
        return None, None
    models = {
        "summarizer_llm": ChatOpenAI(model=CONFIG["summarizer_model"], temperature=0.2, api_key=keys["OPENAI_API_KEY"]),
        "analyst_llm": ChatGoogleGenerativeAI(model=CONFIG["analyst_model"], temperature=0.0, model_kwargs={"response_mime_type": "application/json"}, api_key=keys["GOOGLE_API_KEY"]),
        "sentiment_analyzer": HuggingFaceEndpoint(repo_id=CONFIG["sentiment_model_repo_id"], task="text-classification", huggingfacehub_api_token=keys["HUGGINGFACEHUB_API_TOKEN"])
    }
    return keys, models

# -------- STATE & PROMPTS (For a single ticker) --------
class PipelineState(Dict):
    ticker: str; start_date: dt.date; end_date: dt.date; news_summaries: List[Dict] = []; finance_analysis: Dict = {}; sentiment_scored: List[Dict] = []; report: Dict = {}; error: str = None
summary_prompt = ChatPromptTemplate.from_messages([("system", "You are FinancialNewsSummarizerGPT. Produce exactly 3 concise bullet points preserving numbers and dates."), ("human", "{article_text}")])
aggregate_prompt = ChatPromptTemplate.from_messages([("system", "You are ChiefInvestmentStrategistGPT. For the stock ticker provided, combine the sentiment and price data to create a final investment score from 1 (strong sell) to 10 (strong buy). Explain your reasoning clearly and concisely. Return a single JSON object with keys 'ticker', 'score', and 'reasoning'."), ("human", "Ticker: {ticker}\nSentiment Data:\n{sent}\n\nFinancial Data:\n{fin}")])

# -------- NODE HELPERS --------
def fetch_and_summarise(state: PipelineState, summary_chain) -> PipelineState:
    try:
        t = state["ticker"]
        serper = GoogleSerperAPIWrapper(type_="news")
        query = f"\"{t}\" stock news after:{state['start_date']:%Y-%m-%d} before:{state['end_date']:%Y-%m-%d}"
        result = serper.run(query)
        articles = json.loads(result).get("news", [])
        batch_inputs = [{"article_text": (art.get("snippet") or art.get("title", ""))} for art in articles[:CONFIG["max_articles_per_ticker"]] if (art.get("snippet") or art.get("title"))]
        if not batch_inputs: return state
        batch_results = summary_chain.batch(batch_inputs, {"max_concurrency": 5})
        state["news_summaries"] = [{"summary": summary_text, "link": articles[i].get("link"), "published": articles[i].get("date")} for i, summary_text in enumerate(batch_results)]
    except Exception as e: state["error"] = f"News fetching/summarizing failed: {e}"
    return state

def score_sentiment(state: PipelineState, sentiment_analyzer) -> PipelineState:
    if state.get("error") or not state.get("news_summaries"): return state
    try:
        texts_to_score = [it["summary"] for it in state["news_summaries"] if it.get("summary")]
        if not texts_to_score: return state
        api_results = sentiment_analyzer.batch(texts_to_score)
        for i, item in enumerate(state["news_summaries"]):
            if i < len(api_results):
                pmap = {d["label"].lower(): d["score"] for d in api_results[i]}
                item["sentiment_score"] = round(pmap.get("positive", 0) - pmap.get("negative", 0), 3)
        state["sentiment_scored"] = state["news_summaries"]
    except Exception as e: state["error"] = f"Sentiment analysis failed: {e}"
    return state

def fetch_and_analyse_finance(state: PipelineState) -> PipelineState:
    if state.get("error"): return state
    try:
        t = state["ticker"]
        end_date = state["end_date"] + dt.timedelta(days=1)
        df = yf.download(t, start=state["start_date"], end=end_date, progress=False, ignore_tz=True)
        if df.empty or len(df) < 2: raise ValueError("Not enough historical data found.")
        state["prices_raw"] = df
        n_day_return = (df['Close'].iloc[-1] / df['Close'].iloc[0]) - 1
        largest_move = df['Close'].pct_change().abs().max()
        state["finance_analysis"] = {"period_return_pct": round(n_day_return * 100, 2), "largest_daily_move_pct": round(largest_move * 100, 2)}
    except Exception as e: state["error"] = f"Financial data analysis failed: {e}"
    return state

def aggregate(state: PipelineState, aggregate_chain) -> PipelineState:
    if state.get("error"): return state
    try:
        report = aggregate_chain.invoke({
            "ticker": state["ticker"],
            "sent": json.dumps(state.get("sentiment_scored", "No sentiment data")),
            "fin": json.dumps(state.get("finance_analysis", "No financial data"))
        })
        state["report"] = report
    except Exception as e: state["error"] = f"Final report generation failed: {e}"
    return state

# ==============================================================================
# 2. STREAMLIT DASHBOARD UI (Polished and Bug-Free)
# ==============================================================================
st.set_page_config(layout="wide", page_title="Financial Sentiment Dashboard")
st.title("📈 Financial News & Sentiment Analysis")
st.markdown("An interactive dashboard using a multi-agent LLM pipeline to analyze financial news.")

api_keys, models = load_models_and_keys()
if not models: st.stop()

# --- Simplified Pipeline for a SINGLE Ticker ---
@st.cache_data(show_spinner=False)
def run_pipeline_for_one_ticker(_models, _ticker, _start_date, _end_date):
    summary_chain = summary_prompt | _models['summarizer_llm'] | StrOutputParser()
    aggregate_chain = aggregate_prompt | _models['analyst_llm'] | JsonOutputParser()
    g = StateGraph(PipelineState)
    g.add_node("fetch_and_summarise", lambda state: fetch_and_summarise(state, summary_chain))
    g.add_node("score_sentiment", lambda state: score_sentiment(state, _models['sentiment_analyzer']))
    g.add_node("fetch_and_analyse_finance", fetch_and_analyse_finance)
    g.add_node("aggregate", lambda state: aggregate(state, aggregate_chain))
    g.set_entry_point("fetch_and_summarise")
    g.add_edge("fetch_and_summarise", "score_sentiment")
    g.add_edge("score_sentiment", "fetch_and_analyse_finance")
    g.add_edge("fetch_and_analyse_finance", "aggregate")
    g.add_edge("aggregate", END)
    pipeline = g.compile()
    initial_state = {"ticker": _ticker, "start_date": _start_date, "end_date": _end_date}
    return pipeline.invoke(initial_state)

# --- Sidebar UI ---
st.sidebar.header("Analysis Configuration")
if 'available_tickers' not in st.session_state:
    st.session_state.available_tickers = ['NVDA', 'GOOGL', 'MSFT', 'AAPL']
new_ticker = st.sidebar.text_input("Add Ticker Symbol", placeholder="e.g., CRM").strip().upper()
if st.sidebar.button("Add Ticker"):
    if new_ticker and new_ticker not in st.session_state.available_tickers:
        st.session_state.available_tickers.append(new_ticker)
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
        # --- FIX: Clear previous results to prevent duplicates ---
        st.session_state.all_results = []
        
        results = []
        status_container = st.empty()
        progress_bar = st.progress(0)
        for i, ticker in enumerate(selected_tickers):
            status_container.info(f"▶️ Now analyzing: **{ticker}** ({i+1} of {len(selected_tickers)})...")
            result = run_pipeline_for_one_ticker(models, ticker, start_date, end_date)
            results.append(result)
            progress_bar.progress((i + 1) / len(selected_tickers))
        
        status_container.success("✅ Analysis Complete!")
        st.session_state.all_results = results

# --- Display Results ---
if 'all_results' in st.session_state and st.session_state.all_results:
    results = st.session_state.all_results
    
    st.subheader("Executive Summary Report")
    summary_data = []
    successful_results = []
    for res in results:
        if res.get('error'):
             summary_data.append({"Ticker": res.get('ticker'), "Score": "N/A", "Reasoning": f"ERROR: {res['error']}"})
        elif res.get('report'):
            report = res['report']
            summary_data.append({"Ticker": report.get('ticker'), "Score": report.get('score'), "Reasoning": report.get('reasoning')})
            successful_results.append(res) # Only add successful runs to be displayed in tabs
    
    summary_df = pd.DataFrame(summary_data)
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    if successful_results:
        st.subheader("Detailed Analysis by Ticker")
        tab_titles = [res.get("ticker") for res in successful_results]
        tabs = st.tabs(tab_titles)
        
        for i, res in enumerate(successful_results):
            with tabs[i]:
                st.subheader(f"Financial Overview for {res['ticker']}")
                fin_analysis = res.get("finance_analysis", {})
                col1, col2 = st.columns(2)
                col1.metric(label=f"Return ({start_date} to {end_date})", value=f"{fin_analysis.get('period_return_pct', 0):.2f}%")
                col2.metric(label="Largest Single-Day Move", value=f"{fin_analysis.get('largest_daily_move_pct', 0):.2f}%")
                
                prices_df = res.get("prices_raw")
                if prices_df is not None and not prices_df.empty:
                    fig = px.line(prices_df, y="Close", title=f"{res['ticker']} Stock Price", labels={"Date": "Date", "Close": "Closing Price (USD)"})
                    st.plotly_chart(fig, use_container_width=True)

                st.subheader("News Summaries & Sentiment Scores")
                sentiment_data = res.get("sentiment_scored", [])
                if not sentiment_data: st.info(f"No news articles found or processed.")
                for item in sentiment_data:
                    score = item.get('sentiment_score', 0)
                    color = "green" if score > 0 else "red" if score < 0 else "blue"
                    with st.expander(f"**{item.get('published', 'Date N/A')}** | Score: :{color}[{score:.3f}]"):
                        st.markdown(item.get("summary", "No summary available."))
                        st.link_button("Go to Article", item.get("link", "#"))
else:
    st.info("Configure your analysis in the sidebar and click 'Run Analysis' to begin.")
