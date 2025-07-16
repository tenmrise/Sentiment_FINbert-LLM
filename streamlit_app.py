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
# 1. CORE PIPELINE LOGIC (RE-ARCHITECTED FOR SIMPLICITY)
# ==============================================================================

# --------  CONFIG  --------
CONFIG = {
    "max_articles_per_ticker": 5, # Can increase this later
    "summarizer_model": "gpt-4o",
    "verifier_model": "gpt-4o",
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
        st.error("API Key Missing! Ensure SERPER, OPENAI, GOOGLE, and HUGGINGFACEHUB keys are in secrets.")
        return None, None
    models = {
        "summarizer_llm": ChatOpenAI(model=CONFIG["summarizer_model"], temperature=0.2, api_key=keys["OPENAI_API_KEY"]),
        "verifier_llm": ChatOpenAI(model=CONFIG["verifier_model"], temperature=0.0, api_key=keys["OPENAI_API_KEY"]),
        "analyst_llm": ChatGoogleGenerativeAI(model=CONFIG["analyst_model"], temperature=0.0, model_kwargs={"response_mime_type": "application/json"}, api_key=keys["GOOGLE_API_KEY"]),
        "sentiment_analyzer": HuggingFaceEndpoint(repo_id=CONFIG["sentiment_model_repo_id"], task="text-classification", huggingfacehub_api_token=keys["HUGGINGFACEHUB_API_TOKEN"])
    }
    return keys, models

# -------- STATE (Now simpler, for a single ticker) --------
class PipelineState(Dict):
    ticker: str
    start_date: dt.date
    end_date: dt.date
    news_raw: List[Dict]
    news_summaries: List[Dict]
    sentiment_scored: List[Dict]
    prices_raw: pd.DataFrame
    finance_analysis: Dict
    verification: str | Dict
    report: Dict
    error: str = None

# -------- PROMPTS (Slightly simplified) --------
summary_prompt = ChatPromptTemplate.from_messages([("system", "You are FinancialNewsSummarizerGPT. Produce exactly 3 concise bullet points preserving numbers and dates."), ("human", "{article_text}")])
aggregate_prompt = ChatPromptTemplate.from_messages([("system", "You are ChiefInvestmentStrategistGPT. For the stock ticker provided, combine the sentiment and price data to create a final investment score from 1 (strong sell) to 10 (strong buy). Explain your reasoning. Return a single JSON object with keys 'ticker', 'score', and 'reasoning'."), ("human", "Ticker: {ticker}\nSentiment Data:\n{sent}\n\nFinancial Data:\n{fin}")])

# -------- NODE HELPERS (Now operate on a single ticker) --------
def fetch_news(state: PipelineState) -> PipelineState:
    try:
        t = state["ticker"]
        serper = GoogleSerperAPIWrapper(type_="news")
        query = f"\"{t}\" stock news after:{state['start_date']:%Y-%m-%d} before:{state['end_date']:%Y-%m-%d}"
        result = serper.run(query)
        if isinstance(result, str) and "error" in result.lower(): raise Exception(f"Serper API error for {t}: {result}")
        state["news_raw"] = json.loads(result).get("news", [])
    except Exception as e: state["error"] = f"Failed to fetch news: {e}"
    return state

def summarise_news(state: PipelineState, summary_chain) -> PipelineState:
    if state.get("error"): return state
    try:
        articles = state["news_raw"]
        batch_inputs = [{"article_text": (art.get("snippet") or art.get("title", ""))} for art in articles[:CONFIG["max_articles_per_ticker"]] if (art.get("snippet") or art.get("title"))]
        if not batch_inputs: state["news_summaries"] = []; return state
        batch_results = summary_chain.batch(batch_inputs, {"max_concurrency": 5})
        state["news_summaries"] = [{"summary": summary_text, "link": articles[i].get("link"), "published": articles[i].get("date")} for i, summary_text in enumerate(batch_results)]
    except Exception as e: state["error"] = f"Failed to summarize news: {e}"
    return state
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
# 1. CORE PIPELINE LOGIC
# ==============================================================================

# --------  CONFIG  --------
CONFIG = {
    "max_articles_per_ticker": 5, # Reduced for faster sequential runs
    "gap_threshold_pct": 2.0,
    "summarizer_model": "gpt-4o",
    "verifier_model": "gpt-4o",
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
        "verifier_llm": ChatOpenAI(model=CONFIG["verifier_model"], temperature=0.0, api_key=keys["OPENAI_API_KEY"]),
        "analyst_llm": ChatGoogleGenerativeAI(model=CONFIG["analyst_model"], temperature=0.0, model_kwargs={"response_mime_type": "application/json"}, api_key=keys["GOOGLE_API_KEY"]),
        "sentiment_analyzer": HuggingFaceEndpoint(repo_id=CONFIG["sentiment_model_repo_id"], task="text-classification", huggingfacehub_api_token=keys["HUGGINGFACEHUB_API_TOKEN"])
    }
    return keys, models

# -------- STATE, PROMPTS, etc. --------
class PipelineState(Dict):
    ticker: str; start: dt.date; end: dt.date; news_raw: List[Dict]; news_summaries: List[Dict]; sentiment_scored: List[Dict]; prices_raw: pd.DataFrame; finance_analysis: Dict; verification: str | Dict; report: str; errors: List[str]
summary_prompt = ChatPromptTemplate.from_messages([("system", "You are FinancialNewsSummarizerGPT. Produce exactly 3 concise bullet points preserving numbers and dates."), ("human", "{article_text}")])
verify_prompt = ChatPromptTemplate.from_messages([("system", "You are FactCheckGPT. Cross-reference financial analysis and news sentiment. If consistent, output 'OK'. Else, provide a brief JSON list of observations."), ("human", "Sentiment Analysis:\n```{sent_json}```\n\nFinancial Data:\n```{fin_json}```")])
aggregate_prompt = ChatPromptTemplate.from_messages([("system", "You are ChiefInvestmentStrategistGPT. Combine sentiment and price data for the ticker {ticker} to create an investment score from 1-10. Explain your reasoning. Return a single JSON object."), ("human", "Ticker: {ticker}\nSentiment Data:\n{sent}\n\nFinancial Data:\n{fin}\n\nVerification Notes:\n{corr}")])

# -------- NODE HELPERS (Simplified for single ticker) --------
def fetch_news(state: PipelineState) -> PipelineState:
    t = state["ticker"]
    try:
        serper = GoogleSerperAPIWrapper(type_="news")
        query = f"\"{t}\" stock news after:{state['start']:%Y-%m-%d} before:{state['end']:%Y-%m-%d}"
        result = serper.run(query)
        if isinstance(result, str) and "error" in result.lower(): raise Exception(f"Serper API error for {t}: {result}")
        state["news_raw"] = json.loads(result).get("news", [])
    except Exception as e: state["errors"].append(f"Failed to fetch news for {t}: {e}"); state["news_raw"] = []
    return state
def summarise_news(state: PipelineState, summary_chain) -> PipelineState:
    articles = state["news_raw"]; max_articles = CONFIG["max_articles_per_ticker"]
    if not articles: state["news_summaries"] = []; return state
    batch_inputs = [{"article_text": (art.get("snippet") or art.get("title", ""))} for art in articles[:max_articles] if (art.get("snippet") or art.get("title"))]
    if not batch_inputs: state["news_summaries"] = []; return state
    try:
        batch_results = summary_chain.batch(batch_inputs, {"max_concurrency": 5})
        state["news_summaries"] = [{"summary": summary_text, "link": articles[i].get("link"), "published": articles[i].get("date")} for i, summary_text in enumerate(batch_results)]
    except Exception as e: state["errors"].append(f"Failed to summarize news for {state['ticker']}: {e}"); state["news_summaries"] = []
    return state
def score_sentiment(state: PipelineState, sentiment_analyzer) -> PipelineState:
    items = state["news_summaries"]; texts_to_score = [it["summary"] for it in items if it.get("summary")]
    if not texts_to_score: state["sentiment_scored"] = []; return state
    try:
        api_results = sentiment_analyzer.batch(texts_to_score)
        scored_items = []
        for i, item in enumerate(items):
            if i < len(api_results):
                pmap = {d["label"].lower(): d["score"] for d in api_results[i]}
                score = pmap.get("positive", 0) - pmap.get("negative", 0)
                scored_items.append({**item, "sentiment_score": round(score, 3)})
        state["sentiment_scored"] = scored_items
    except Exception as e: state["errors"].append(f"Failed to score sentiment for {state['ticker']} via API: {e}"); state["sentiment_scored"] = []
    return state
def fetch_finance_data(state: PipelineState) -> PipelineState:
    t = state["ticker"]
    try:
        end_date = state["end"] + dt.timedelta(days=1); df = yf.download(t, start=state["start"], end=end_date, progress=False)
        if df.empty: raise ValueError("No data returned from Yahoo Finance.")
        state["prices_raw"] = df
    except Exception as e: state["errors"].append(f"Failed to download financial data for {t}: {e}"); state["prices_raw"] = pd.DataFrame()
    return state
def analyse_finance_data(state: PipelineState) -> PipelineState:
    df = state["prices_raw"]
    if df.empty: state["finance_analysis"] = {"error": "Missing price data"}; return state
    if len(df) < 2: state["finance_analysis"] = {"error": "Not enough data for analysis."}; return state
    n_day_return = (df['Close'].iloc[-1] / df['Close'].iloc[0]) - 1; largest_move = df['Close'].pct_change().abs().max()
    state["finance_analysis"] = {"period_return_pct": round(n_day_return * 100, 2), "largest_daily_move_pct": round(largest_move * 100, 2)}
    return state

def score_sentiment(state: PipelineState, sentiment_analyzer) -> PipelineState:
    if state.get("error"): return state
    try:
        items = state["news_summaries"]
        texts_to_score = [it["summary"] for it in items if it.get("summary")]
        if not texts_to_score: state["sentiment_scored"] = []; return state
        api_results = sentiment_analyzer.batch(texts_to_score)
        scored_items = []
        for i, item in enumerate(items):
            if i < len(api_results):
                pmap = {d["label"].lower(): d["score"] for d in api_results[i]}
                score = pmap.get("positive", 0) - pmap.get("negative", 0)
                scored_items.append({**item, "sentiment_score": round(score, 3)})
        state["sentiment_scored"] = scored_items
    except Exception as e: state["error"] = f"Failed to score sentiment via API: {e}"
    return state

def fetch_and_analyse_finance(state: PipelineState) -> PipelineState:
    if state.get("error"): return state
    try:
        t = state["ticker"]
        end_date = state["end_date"] + dt.timedelta(days=1)
        df = yf.download(t, start=state["start_date"], end=end_date, progress=False)
        if df.empty: raise ValueError("No data returned from Yahoo Finance.")
        state["prices_raw"] = df
        
        if len(df) < 2: raise ValueError("Not enough data for analysis.")
        n_day_return = (df['Close'].iloc[-1] / df['Close'].iloc[0]) - 1
        largest_move = df['Close'].pct_change().abs().max()
        state["finance_analysis"] = {"period_return_pct": round(n_day_return * 100, 2), "largest_daily_move_pct": round(largest_move * 100, 2)}
    except Exception as e: state["error"] = f"Finance analysis failed: {e}"
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
# 2. STREAMLIT DASHBOARD UI
# ==============================================================================
st.set_page_config(layout="wide", page_title="Financial Sentiment Dashboard")
st.title("Financial News & Sentiment Analysis Dashboard")
st.markdown("An interactive dashboard using a multi-agent LLM pipeline to analyze financial news.")

api_keys, models = load_models_and_keys()
if not models: st.stop()

summary_chain = summary_prompt | models['summarizer_llm'] | StrOutputParser()
aggregate_chain = aggregate_prompt | models['analyst_llm'] | JsonOutputParser()

# --- Simplified Pipeline for a SINGLE Ticker ---
@st.cache_data(show_spinner=False)
def run_pipeline_for_one_ticker(_ticker, _start_date, _end_date):
    g = StateGraph(PipelineState)
    g.add_node("fetch_news", fetch_news)
    g.add_node("summarise_news", lambda state: summarise_news(state, summary_chain))
    g.add_node("score_sentiment", lambda state: score_sentiment(state, models['sentiment_analyzer']))
    g.add_node("fetch_and_analyse_finance", fetch_and_analyse_finance)
    g.add_node("aggregate", lambda state: aggregate(state, aggregate_chain))

    # A simple, linear graph
    g.set_entry_point("fetch_news")
    g.add_edge("fetch_news", "summarise_news")
    g.add_edge("summarise_news", "score_sentiment")
    g.add_edge("score_sentiment", "fetch_and_analyse_finance")
    g.add_edge("fetch_and_analyse_finance", "aggregate")
    g.add_edge("aggregate", END)
    
    pipeline = g.compile()
    initial_state = {"ticker": _ticker, "start_date": _start_date, "end_date": _end_date}
    return pipeline.invoke(initial_state)

# --- Sidebar UI ---
st.sidebar.header("Analysis Configuration")
if 'available_tickers' not in st.session_state: st.session_state.available_tickers = ['NVDA', 'GOOGL', 'MSFT', 'AAPL']
new_ticker = st.sidebar.text_input("Add Ticker Symbol", placeholder="e.g., CRM").strip().upper()
if st.sidebar.button("Add Ticker"):
    if new_ticker and new_ticker not in st.session_state.available_tickers:
        st.session_state.available_tickers.append(new_ticker)
    elif not new_ticker:
        st.sidebar.warning("Please enter a ticker symbol.")
    else:
        st.sidebar.warning(f"{new_ticker} is already in the list.")

def crosscheck(state: PipelineState, verify_chain) -> PipelineState:
    try:
        result = verify_chain.invoke({"sent_json": json.dumps(state["sentiment_scored"]), "fin_json": json.dumps(state["finance_analysis"])})
        try: state["verification"] = json.loads(result)
        except json.JSONDecodeError: state["verification"] = result
    except Exception as e: state["errors"].append(f"Verification step failed for {state['ticker']}: {e}"); state["verification"] = "Verification failed"
    return state
def aggregate(state: PipelineState, aggregate_chain) -> PipelineState:
    try:
        report = aggregate_chain.invoke({"ticker": state["ticker"], "sent": json.dumps(state["sentiment_scored"]), "fin": json.dumps(state["finance_analysis"]), "corr": json.dumps(state["verification"])})
        state["report"] = report
    except Exception as e: state["errors"].append(f"Final aggregation step failed for {state['ticker']}: {e}"); state["report"] = {"error": "Report generation failed."}
    return state

# ==============================================================================
# 2. STREAMLIT DASHBOARD UI
# ==============================================================================
st.set_page_config(layout="wide", page_title="Financial Sentiment Dashboard")
st.title("Financial News & Sentiment Analysis Dashboard")
st.markdown("An interactive dashboard using a multi-agent LLM pipeline to analyze financial news.")

api_keys, models = load_models_and_keys()
if not models: st.stop()

summary_chain = summary_prompt | models['summarizer_llm'] | StrOutputParser()
verify_chain = verify_prompt | models['verifier_llm'] | StrOutputParser()
aggregate_chain = aggregate_prompt | models['analyst_llm'] | JsonOutputParser()

# --- The pipeline function now only needs to be defined once ---
@st.cache_resource
def build_pipeline():
    g = StateGraph(PipelineState)
    g.add_node("fetch_news", fetch_news)
    g.add_node("summarise_news", lambda state: summarise_news(state, summary_chain))
    g.add_node("score_sentiment", lambda state: score_sentiment(state, models['sentiment_analyzer']))
    g.add_node("fetch_finance", fetch_finance_data)
    g.add_node("analyse_finance", lambda state: analyse_finance_data(state))
    g.add_node("crosscheck", lambda state: crosscheck(state, verify_chain))
    g.add_node("aggregate", lambda state: aggregate(state, aggregate_chain))
    g.set_entry_point("fetch_news")
    # A simple, robust, sequential graph is now possible and much safer
    g.add_edge("fetch_news", "summarise_news")
    g.add_edge("summarise_news", "score_sentiment")
    g.add_edge("score_sentiment", "fetch_finance")
    g.add_edge("fetch_finance", "analyse_finance")
    g.add_edge("analyse_finance", "crosscheck")
    g.add_edge("crosscheck", "aggregate")
    g.add_edge("aggregate", END)
    return g.compile()

pipeline = build_pipeline()

# --- Main app logic ---
st.sidebar.header("Analysis Configuration")
if 'available_tickers' not in st.session_state:
    st.session_state.available_tickers = ['NVDA', 'GOOGL', 'MSFT', 'AAPL']
new_ticker = st.sidebar.text_input("Add Ticker Symbol", placeholder="e.g., CRM").strip().upper()
if st.sidebar.button("Add Ticker"):
    if new_ticker and new_ticker not in st.session_state.available_tickers:
        st.session_state.available_tickers.append(new_ticker)
    elif not new_ticker:
        st.sidebar.warning("Please enter a ticker symbol.")
    else:
        st.sidebar.warning(f"{new_ticker} is already in the list.")
selected_tickers = st.sidebar.multiselect("Select Stock Tickers for Analysis", options=st.session_state.available_tickers, default=['NVDA', 'MSFT'])
today = dt.date.today()
start_date = st.sidebar.date_input("Start Date", value=today - dt.timedelta(days=7))
end_date = st.sidebar.date_input("End Date", value=today)

if st.sidebar.button("🚀 Run Analysis", type="primary"):
    if not selected_tickers: st.warning("Please select at least one ticker.")
    elif start_date >= end_date: st.warning("Start Date must be before End Date.")
    else:
        # --- Main UI Loop: Run pipeline for each ticker sequentially ---
        all_results = []
        progress_bar = st.progress(0, text="Starting Analysis...")
        for i, ticker in enumerate(selected_tickers):
            progress_bar.progress((i) / len(selected_tickers), text=f"Analyzing {ticker}...")
            result = run_pipeline_for_one_ticker(ticker, start_date, end_date)
            all_results.append(result)
        progress_bar.progress(1.0, text="Analysis Complete!")
        st.session_state.all_results = all_results

# --- Display Results ---
if 'all_results' in st.session_state:
    results = st.session_state.all_results
    
    # --- Summary Tab ---
    st.header("📈 Summary Report")
    summary_data = []
    for res in results:
        if res.get('report'):
            summary_data.append(res['report'])
        elif res.get('error'):
             summary_data.append({"ticker": res.get('ticker'), "score": "N/A", "reasoning": f"ERROR: {res['error']}"})
    st.table(summary_data)

    # --- Ticker Detail Tabs ---
    st.header("📄 Detailed Analysis")
    tab_titles = [res.get("ticker", "Error") for res in results]
    tabs = st.tabs(tab_titles)
    
    for i, res in enumerate(results):
        with tabs[i]:
            if res.get("error"):
                st.error(f"Could not complete analysis for {res.get('ticker')}: {res.get('error')}")
                continue

            st.subheader("Financial Overview")
            fin_analysis = res.get("finance_analysis", {})
            col1, col2 = st.columns(2)
            col1.metric(label=f"Return ({start_date} to {end_date})", value=f"{fin_analysis.get('period_return_pct', 0)}%")
            col2.metric(label="Largest Single-Day Move", value=f"{fin_analysis.get('largest_daily_move_pct', 0)}%")
            prices_df = res.get("prices_raw")
            if prices_df is not None and not prices_df.empty:
                fig = px.line(prices_df, y="Close", title=f"{res['ticker']} Stock Price", labels={"Date": "Date", "Close": "Closing Price (USD)"})
                st.plotly_chart(fig, use_container_width=True)

            st.subheader("News Sentiment Analysis")
            sentiment_data = res.get("sentiment_scored", [])
            if not sentiment_data: st.info(f"No news articles found or processed.")
            for item in sentiment_data:
                with st.expander(f"**{item.get('published', 'Date N/A')}** | Score: {item.get('sentiment_score', 0)}"):
                    st.markdown(item.get("summary", "No summary available."))
                    st.link_button("Go to Article", item.get("link", "#"))
else:
    st.info("Configure your analysis in the sidebar and click 'Run Analysis' to begin.")
