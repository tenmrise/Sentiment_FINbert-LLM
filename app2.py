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
# 1. CORE PIPELINE LOGIC (Linear, Uncontaminated, and Robust)
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
    keys = {"SERPER_API_KEY": os.getenv("SERPER_API_KEY"),"OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),"GOOGLE_API_KEY": os.getenv("GOOGLE_API_KEY"),"HUGGINGFACEHUB_API_TOKEN": os.getenv("HUGGINGFACEHUB_API_TOKEN")}
    if not all(keys.values()):
        st.error("API Key Missing! Ensure SERPER, OPENAI, GOOGLE, and HUGGINGFACEHUB keys are in your secrets.")
        return None, None
    models = {
        "fact_extractor_llm": ChatOpenAI(model=CONFIG["extractor_model"], temperature=0.0, api_key=keys["OPENAI_API_KEY"]),
        "analyst_llm": ChatGoogleGenerativeAI(model=CONFIG["analyst_model"], temperature=0.1, model_kwargs={"response_mime_type": "application/json"}, api_key=keys["GOOGLE_API_KEY"]),
        "sentiment_analyzer": HuggingFaceEndpoint(repo_id=CONFIG["sentiment_model_repo_id"], task="text-classification", huggingfacehub_api_token=keys["HUGGINGFACEHUB_API_TOKEN"])
    }
    return keys, models

# -------- STATE & PROMPTS --------
class PipelineState(Dict):
    ticker: str; start_date: dt.date; end_date: dt.date; news_raw: List[Dict] = []; key_facts: List[Dict] = []; finance_analysis: Dict = {}; sentiment_scored: List[Dict] = []; sentiment_analysis: Dict = {}; report: Dict = {}; error: str = None

fact_extraction_prompt = ChatPromptTemplate.from_messages([("system", "You are a data extraction engine. From the provided news article text, extract the following information. Do not interpret, analyze, or add any information not present in the text. Your output must be a JSON object with the keys 'key_figures', 'core_event', and 'outlook'. If a key is not mentioned in the text, its value should be 'Not mentioned'."),("human", "Article Text:\n```{article_text}```")])
aggregate_prompt = ChatPromptTemplate.from_messages([("system", "You are a quantitative investment strategist. Your task is to synthesize three independent streams of pre-computed data: quantitative sentiment, extracted factual catalysts, and market price action. Your goal is to identify **divergence** or **convergence** between the news narrative and the stock's performance.\n\n**Analytical Framework:**\n1.  **Review Sentiment Profile:** Is the statistical sentiment profile Positive, Negative, or Contentious (high standard deviation)?\n2.  **Review Factual Catalysts:** Do the extracted key facts represent clear positive or negative events?\n3.  **Review Price Action:** Did the stock significantly outperform, underperform, or track the market?\n4.  **Formulate Thesis:** Synthesize the three data streams. Is there a clear DIVERGENCE? (e.g., 'Despite a negative sentiment profile and no clear positive catalysts, the stock remained resilient, suggesting the market has already priced in known risks.') Or is there a CONVERGENCE? (e.g., 'Strong positive sentiment, driven by the new product launch, is confirmed by the stock's significant outperformance.')\n\n**Output Schema (Strict JSON):**\n```json\n{\n  \"ticker\": \"<The stock ticker>\",\n  \"investment_thesis\": \"<Your concise thesis based on the divergence/convergence analysis>\",\n  \"final_score\": <Integer from 1 to 10>,\n  \"score_justification\": \"<1-sentence justification for your score, citing the thesis.>\"\n}\n```"),("human","Input Data for Ticker: {ticker}\n\n**Quantitative Sentiment Profile:**\n```{sent_analysis}```\n\n**Extracted Factual Catalysts:**\n```{key_facts}```\n\n**Quantitative Price Action:**\n```{fin_analysis}```")])

# -------- NODE HELPERS --------
def fetch_news(state: PipelineState) -> PipelineState:
    try:
        t = state["ticker"]; serper = GoogleSerperAPIWrapper(type_="news")
        query = f"\"{t}\" stock news after:{state['start_date']:%Y-%m-%d} before:{state['end_date']:%Y-%m-%d}"
        result = serper.run(query)
        state["news_raw"] = json.loads(result).get("news", [])[:CONFIG["max_articles_per_ticker"]]
    except Exception as e: state["error"] = f"News fetching failed: {e}"
    return state

def extract_key_facts(state: PipelineState, fact_extraction_chain) -> PipelineState:
    if state.get("error") or not state.get("news_raw"): return state
    try:
        articles = state["news_raw"]
        batch_inputs = [{"article_text": (art.get("snippet") or art.get("title", ""))} for art in articles]
        if not batch_inputs: return state
        batch_results = fact_extraction_chain.batch(batch_inputs, {"max_concurrency": 5})
        for i, article in enumerate(articles):
            if i < len(batch_results): article["key_facts"] = batch_results[i]
        state["key_facts"] = articles
    except Exception as e: state["error"] = f"Fact extraction failed: {e}"
    return state

def score_sentiment(state: PipelineState, sentiment_analyzer) -> PipelineState:
    if state.get("error") or not state.get("news_raw"): return state
    try:
        articles = state["news_raw"]
        texts_to_score = [(art.get("snippet") or art.get("title", "")) for art in articles]
        if not texts_to_score: return state
        api_results = sentiment_analyzer.batch(texts_to_score)
        scores = []
        for i, article in enumerate(articles):
            if i < len(api_results):
                pmap = {d["label"].lower(): d["score"] for d in api_results[i]}
                score = round(pmap.get("positive", 0) - pmap.get("negative", 0), 3)
                article["sentiment_score"] = score; scores.append(score)
        state["sentiment_scored"] = articles
        if scores:
            sdf = pd.DataFrame(scores, columns=["score"])
            state["sentiment_analysis"] = {"average_score": round(sdf["score"].mean(), 3), "std_dev_sentiment": round(sdf["score"].std(), 3) if len(scores) > 1 else 0, "num_articles": len(scores), "num_positive": int(sdf[sdf["score"] > 0.1].count().iloc[0]), "num_negative": int(sdf[sdf["score"] < -0.1].count().iloc[0]), "num_neutral": int(sdf[(sdf["score"] >= -0.1) & (sdf["score"] <= 0.1)].count().iloc[0])}
    except Exception as e: state["error"] = f"Sentiment analysis failed: {e}"
    return state

def fetch_and_analyse_finance(state: PipelineState) -> PipelineState:
    if state.get("error"): return state
    try:
        t = state["ticker"]; end_date = state["end_date"] + dt.timedelta(days=1)
        df = yf.download(t, start=state["start_date"], end=end_date, progress=False, ignore_tz=True)
        if df.empty or len(df) < 2: raise ValueError("Not enough historical data found.")
        state["prices_raw"] = df
        n_day_return = (df['Close'].iloc[-1] / df['Close'].iloc[0]) - 1; largest_move = df['Close'].pct_change().abs().max()
        state["finance_analysis"] = {"period_return_pct": round(n_day_return * 100, 2), "largest_daily_move_pct": round(largest_move * 100, 2)}
    except Exception as e: state["error"] = f"Financial data analysis failed: {e}"
    return state

def aggregate(state: PipelineState, aggregate_chain) -> PipelineState:
    if state.get("error"): return state
    try:
        key_facts_for_prompt = [a.get("key_facts", {}) for a in state.get("key_facts", [])]
        report = aggregate_chain.invoke({"ticker": state["ticker"], "sent_analysis": json.dumps(state.get("sentiment_analysis", {})), "key_facts": json.dumps(key_facts_for_prompt), "fin_analysis": json.dumps(state.get("finance_analysis", {}))})
        state["report"] = report
    except Exception as e: state["error"] = f"Final report generation failed: {e}"
    return state

# ==============================================================================
# 2. STREAMLIT DASHBOARD UI
# ==============================================================================
st.set_page_config(layout="wide", page_title="Financial Sentiment Dashboard")
st.title("📈 Financial News & Sentiment Analysis")
st.markdown("A dashboard for quantitative synthesis of news sentiment and price action, powered by LLM agents.")

api_keys, models = load_models_and_keys()
if not models: st.stop()

@st.cache_data(show_spinner=False)
def run_pipeline_for_one_ticker(_models, _ticker, _start_date, _end_date):
    fact_extraction_chain = fact_extraction_prompt | _models['fact_extractor_llm'] | JsonOutputParser()
    aggregate_chain = aggregate_prompt | _models['analyst_llm'] | JsonOutputParser()
    
    # --- FIX: A simple, robust, LINEAR graph. No parallelism. ---
    g = StateGraph(PipelineState)
    g.add_node("fetch_news", fetch_news)
    g.add_node("extract_key_facts", lambda state: extract_key_facts(state, fact_extraction_chain))
    g.add_node("score_sentiment", lambda state: score_sentiment(state, _models['sentiment_analyzer']))
    g.add_node("fetch_and_analyse_finance", fetch_and_analyse_finance)
    g.add_node("aggregate", lambda state: aggregate(state, aggregate_chain))

    g.set_entry_point("fetch_news")
    g.add_edge("fetch_news", "extract_key_facts")
    g.add_edge("extract_key_facts", "score_sentiment")
    g.add_edge("score_sentiment", "fetch_and_analyse_finance")
    g.add_edge("fetch_and_analyse_finance", "aggregate")
    g.add_edge("aggregate", END)

    pipeline = g.compile()
    initial_state = {"ticker": _ticker, "start_date": _start_date, "end_date": _end_date}
    return pipeline.invoke(initial_state)

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
            result = run_pipeline_for_one_ticker(models, ticker, start_date, end_date)
            results.append(result); progress_bar.progress((i + 1) / len(selected_tickers))
        status_container.success("✅ Analysis Complete!"); st.session_state.all_results = results

if 'all_results' in st.session_state and st.session_state.all_results:
    results = st.session_state.all_results
    st.subheader("Executive Summary Report")
    summary_data = []
    successful_results = []
    for res in results:
        if res.get('error'): summary_data.append({"Ticker": res.get('ticker'), "Score": "N/A", "Thesis": f"ERROR: {res['error']}"})
        elif res.get('report'): report = res['report']; summary_data.append({"Ticker": report.get('ticker'), "Score": report.get('final_score'), "Thesis": report.get('investment_thesis')}); successful_results.append(res)
    summary_df = pd.DataFrame(summary_data); st.dataframe(summary_df, use_container_width=True, hide_index=True)

    if successful_results:
        st.subheader("Detailed Analysis by Ticker"); tab_titles = [res.get("ticker") for res in successful_results]; tabs = st.tabs(tab_titles)
        for i, res in enumerate(successful_results):
            with tabs[i]:
                st.subheader(f"Data Synthesis for {res['ticker']}")
                report = res['report']
                st.info(f"**Investment Thesis:** {report.get('investment_thesis')}")
                st.write(f"**Justification:** {report.get('score_justification')}")
                
                st.subheader("Raw Data & Extracted Facts")
                col1, col2 = st.columns(2); col1.metric(label=f"Return ({start_date} to {end_date})", value=f"{res['finance_analysis'].get('period_return_pct', 0):.2f}%"); col2.metric(label="Largest Single-Day Move", value=f"{res['finance_analysis'].get('largest_daily_move_pct', 0):.2f}%")
                
                prices_df = res.get("prices_raw")
                if prices_df is not None and not prices_df.empty: fig = px.line(prices_df, y="Close", title=f"{res['ticker']} Stock Price"); st.plotly_chart(fig, use_container_width=True)
                
                news_data = res.get("key_facts", []) 
                if not news_data: st.info(f"No news articles found.")
                for item in news_data:
                    score = item.get('sentiment_score', 0); color = "green" if score > 0 else "red" if score < 0 else "blue"
                    with st.expander(f"**{item.get('published', 'Date N/A')}** | Score: :{color}[{score:.3f}]"):
                        st.write("**Key Facts Extracted:**")
                        st.json(item.get('key_facts', 'No facts extracted.'))
                        st.link_button("Go to Article", item.get("link", "#"))
else:
    st.info("Configure your analysis in the sidebar and click 'Run Analysis' to begin.")
