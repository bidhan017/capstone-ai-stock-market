import streamlit as st
import os
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import mlflow.deployments
from databricks.sdk import WorkspaceClient
from databricks.vector_search.client import VectorSearchClient
import yfinance as yf

# Page configuration
st.set_page_config(
    page_title="AI Stock Market Research Assistant",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuration from environment variables
VECTOR_SEARCH_ENDPOINT = os.getenv("VECTOR_SEARCH_ENDPOINT", "vector_search")
VECTOR_INDEX_NAME = os.getenv("VECTOR_INDEX_NAME", "main.stock_research.document_embeddings_index")
EMBEDDING_MODEL_ENDPOINT = os.getenv("EMBEDDING_MODEL_ENDPOINT", "databricks-bge-large-en")

# Initialize clients
w = WorkspaceClient()
vsc = VectorSearchClient()
mlflow_client = mlflow.deployments.get_deploy_client("databricks")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "user_email" not in st.session_state:
    st.session_state.user_email = "demo@example.com"

# Helper functions

def get_embedding(text: str) -> List[float]:
    """Generate embedding for text."""
    response = mlflow_client.predict(
        endpoint=EMBEDDING_MODEL_ENDPOINT,
        inputs={"input": [text[:8000]]}
    )
    return response["data"][0]["embedding"]

def semantic_search(query: str, doc_type: Optional[str] = None, ticker: Optional[str] = None, top_k: int = 5) -> List[Dict]:
    """Search vector index."""
    try:
        query_embedding = get_embedding(query)
        index = vsc.get_index(
            endpoint_name=VECTOR_SEARCH_ENDPOINT,
            index_name=VECTOR_INDEX_NAME
        )
        
        filters = []
        if doc_type:
            filters.append(f"doc_type = '{doc_type}'")
        if ticker:
            filters.append(f"ticker = '{ticker}'")
        
        filter_str = " AND ".join(filters) if filters else None
        
        results = index.similarity_search(
            query_vector=query_embedding,
            columns=["doc_id", "doc_type", "ticker", "title", "text", "metadata_date"],
            filters=filter_str,
            num_results=top_k
        )
        
        docs = []
        for row in results.get("result", {}).get("data_array", []):
            docs.append({
                "doc_id": row[0],
                "doc_type": row[1],
                "ticker": row[2],
                "title": row[3],
                "text": row[4][:500],
                "date": row[5],
                "score": row[-1]
            })
        return docs
    except Exception as e:
        st.error(f"Search error: {e}")
        return []

def get_stock_quote(ticker: str) -> Dict:
    """Get current stock quote using yfinance."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        return {
            "price": info.get("currentPrice", info.get("regularMarketPrice", 0)),
            "change": info.get("regularMarketChange", 0),
            "change_percent": info.get("regularMarketChangePercent", 0),
            "volume": info.get("volume", 0),
            "day_high": info.get("dayHigh", 0),
            "day_low": info.get("dayLow", 0),
            "market_cap": info.get("marketCap", 0)
        }
    except Exception as e:
        return {"error": str(e)}

def get_stock_history(ticker: str, start_date: str, end_date: str) -> Dict:
    """Get historical stock data using yfinance."""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(start=start_date, end=end_date)
        
        if hist.empty:
            return {"error": "No data available"}
        
        prices = []
        for date, row in hist.iterrows():
            prices.append({
                "date": date.strftime("%Y-%m-%d"),
                "open": float(row["Open"]),
                "high": float(row["High"]),
                "low": float(row["Low"]),
                "close": float(row["Close"]),
                "volume": int(row["Volume"])
            })
        
        return {"prices": prices}
    except Exception as e:
        return {"error": str(e)}

# In-memory storage (session state)
def get_watchlist(user_email: str) -> pd.DataFrame:
    if "watchlist" not in st.session_state:
        st.session_state.watchlist = []
    return pd.DataFrame(st.session_state.watchlist, columns=["ticker", "added_at"])

def add_to_watchlist(ticker: str, user_email: str):
    if "watchlist" not in st.session_state:
        st.session_state.watchlist = []
    st.session_state.watchlist.append({"ticker": ticker.upper(), "added_at": datetime.now()})

def remove_from_watchlist(ticker: str, user_email: str):
    if "watchlist" in st.session_state:
        st.session_state.watchlist = [w for w in st.session_state.watchlist if w["ticker"] != ticker.upper()]

# UI Components
def render_sidebar():
    """Render sidebar with watchlist and controls."""
    st.sidebar.title("📊 Your Watchlist")
    
    # Add ticker input
    with st.sidebar.form("add_ticker_form"):
        new_ticker = st.text_input("Add Ticker", placeholder="e.g., AAPL")
        submitted = st.form_submit_button("Add to Watchlist")
        
        if submitted and new_ticker:
            add_to_watchlist(new_ticker, st.session_state.user_email)
            st.success(f"Added {new_ticker.upper()} to watchlist!")
            st.rerun()
    
    # Display watchlist
    watchlist_df = get_watchlist(st.session_state.user_email)
    
    if not watchlist_df.empty:
        st.sidebar.subheader("Your Stocks")
        for _, row in watchlist_df.iterrows():
            col1, col2 = st.sidebar.columns([3, 1])
            with col1:
                if st.button(row['ticker'], key=f"ticker_{row['ticker']}", use_container_width=True):
                    st.session_state.selected_ticker = row['ticker']
                    st.rerun()
            with col2:
                if st.button("❌", key=f"remove_{row['ticker']}"):
                    remove_from_watchlist(row['ticker'], st.session_state.user_email)
                    st.rerun()
    else:
        st.sidebar.info("Your watchlist is empty. Add some tickers!")

def render_stock_details(ticker: str):
    """Render detailed stock information."""
    st.header(f"📈 {ticker} - Stock Details")
    
    # Get current price
    quote = get_stock_quote(ticker)
    
    if "error" not in quote:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Price", f"${quote.get('price', 0):.2f}", 
                     f"{quote.get('change_percent', 0):+.2f}%")
        with col2:
            st.metric("Volume", f"{quote.get('volume', 0):,}")
        with col3:
            st.metric("Day High", f"${quote.get('day_high', 0):.2f}")
        with col4:
            st.metric("Day Low", f"${quote.get('day_low', 0):.2f}")
        
        # Get historical data for chart
        end_date = datetime.now()
        start_date = end_date - timedelta(days=90)
        
        history = get_stock_history(
            ticker,
            start_date.strftime("%Y-%m-%d"),
            end_date.strftime("%Y-%m-%d")
        )
        
        if "error" not in history and history.get('prices'):
            prices = history['prices']
            df = pd.DataFrame(prices)
            df['date'] = pd.to_datetime(df['date'])
            
            # Create candlestick chart
            fig = go.Figure(data=[go.Candlestick(
                x=df['date'],
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close']
            )])
            
            fig.update_layout(
                title=f"{ticker} - 90 Day Price History",
                yaxis_title="Price ($)",
                xaxis_title="Date",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # Show relevant news and context
    st.subheader("📰 Relevant News & Analysis")
    
    search_results = semantic_search(
        f"recent news and analysis for {ticker}",
        ticker=ticker,
        top_k=5
    )
    
    if search_results:
        for doc in search_results:
            with st.expander(f"{doc['title']} ({doc['date']})"):
                st.write(doc['text'])
                st.caption(f"Type: {doc['doc_type']} | Relevance: {doc['score']:.4f}")
    else:
        st.info(f"No news found for {ticker}")
    
    st.info("Note: Watchlist and notes are stored in session (temporary). Connect Lakebase for persistence.")

def render_chat():
    """Render chat interface."""
    st.header("💬 AI Research Assistant")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask about stocks, search for companies, or request analysis..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            response = process_chat_message(prompt)
            st.markdown(response)
        
        st.session_state.messages.append({"role": "assistant", "content": response})

def process_chat_message(message: str) -> str:
    """Process chat message and generate response."""
    message_lower = message.lower()
    
    # Search query
    if "search" in message_lower or "find" in message_lower:
        query = message.replace("search for", "").replace("find", "").strip()
        results = semantic_search(query, top_k=3)
        
        if results:
            response = f"I found {len(results)} relevant results:\n\n"
            for i, doc in enumerate(results, 1):
                response += f"**{i}. [{doc['ticker']}] {doc['title']}**\n"
                response += f"{doc['text'][:200]}...\n\n"
            return response
        else:
            return "I couldn't find any relevant results for that query."
    
    # Price query
    elif "price" in message_lower:
        words = message.split()
        tickers = [w.upper() for w in words if w.isupper() and len(w) <= 5]
        
        if tickers:
            ticker = tickers[0]
            quote = get_stock_quote(ticker)
            
            if "error" not in quote:
                return f"""**{ticker} Current Price:**
                
- Price: ${quote.get('price', 0):.2f}
- Change: {quote.get('change', 0)} ({quote.get('change_percent', 0):+.2f}%)
- Volume: {quote.get('volume', 0):,}
- Market Cap: ${quote.get('market_cap', 0):,}
"""
            else:
                return f"Sorry, I couldn't fetch price data for {ticker}."
    
    # Default response
    return """I can help you with:

- **Search**: "Search for companies in the EV sector"
- **Stock Info**: "What's the price of AAPL?"
- **Analysis**: "Find news about interest rate exposure"
- **Watchlist**: Use the sidebar to manage your watchlist

What would you like to know?"""

# Main app
def main():
    st.title("🤖 AI Stock Market Research Assistant")
    
    # Render sidebar
    render_sidebar()
    
    # Main content area
    if "selected_ticker" in st.session_state and st.session_state.selected_ticker:
        render_stock_details(st.session_state.selected_ticker)
    else:
        render_chat()

if __name__ == "__main__":
    main()
