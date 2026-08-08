import streamlit as st
import os
import sys
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import yfinance as yf
import pickle
from databricks import sql as databricks_sql
import psycopg2
from databricks.sdk import WorkspaceClient

# Import the agent (assumes 04_Agent.py is in same directory or PYTHONPATH)
try:
    # Add current directory to path for imports
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    
    # Import agent runner
    import importlib.util
    spec = importlib.util.spec_from_file_location("agent_module", os.path.join(current_dir, "04_Agent.py"))
    agent_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(agent_module)
    run_agent = agent_module.run_agent
    AGENT_AVAILABLE = True
except Exception as e:
    AGENT_AVAILABLE = False
    print(f"Agent not available: {e}")

# Load resources (cached to avoid reloading)
@st.cache_resource
def load_embedding_model():
    """Load sentence transformer model once."""
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource
def load_search_index():
    """Load pre-computed embeddings from pickle file."""
    try:
        with open('embeddings.pkl', 'rb') as f:
            df = pickle.load(f)
            return df
    except FileNotFoundError:
        # Return empty DataFrame - app will show info message
        return pd.DataFrame()

# Helper functions

def semantic_search(query: str, ticker: Optional[str] = None, top_k: int = 5) -> List[Dict]:
    """Search using local embeddings and cosine similarity."""
    df = load_search_index()
    if df is None or df.empty:
        return []
    
    try:
        model = load_embedding_model()
        query_embedding = model.encode(query)
        
        # Filter by ticker if specified
        search_df = df[df['ticker'] == ticker.upper()] if ticker else df
        
        if search_df.empty:
            return []
        
        # Calculate cosine similarity
        similarities = search_df['embedding'].apply(
            lambda x: np.dot(query_embedding, x) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(x)
            )
        )
        
        # Get top results
        top_indices = similarities.nlargest(top_k).index
        results = search_df.loc[top_indices]
        
        return [{
            'ticker': row['ticker'],
            'title': row.get('title', 'No title'),
            'text': row['text'][:500] + '...' if len(row['text']) > 500 else row['text'],
            'date': row.get('date', 'N/A'),
            'doc_type': row.get('doc_type', 'document'),
            'score': float(similarities[idx])
        } for idx, row in results.iterrows()]
        
    except Exception:
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

# Lakebase configuration
LAKEBASE_PROJECT_NAME = os.getenv("LAKEBASE_PROJECT", "new_database")
LAKEBASE_BRANCH_NAME = os.getenv("LAKEBASE_BRANCH", "production")
LAKEBASE_DATABASE_NAME = os.getenv("LAKEBASE_DATABASE", "databricks_postgres")

@st.cache_resource
def get_lakebase_connection():
    """Get persistent psycopg2 connection to Lakebase Postgres."""
    try:
        w = WorkspaceClient()
        endpoints = list(w.postgres.list_endpoints(
            parent=f"projects/{LAKEBASE_PROJECT_NAME}/branches/{LAKEBASE_BRANCH_NAME}"
        ))
        
        if not endpoints:
            return None
        
        endpoint_path = endpoints[0].name
        token_response = w.postgres.generate_database_credential(endpoint=endpoint_path)
        
        host = f"{LAKEBASE_PROJECT_NAME}-{LAKEBASE_BRANCH_NAME}.cloud.databricks.com"
        
        conn = psycopg2.connect(
            host=host,
            port=5432,
            database=LAKEBASE_DATABASE_NAME,
            user="oauth",
            password=token_response.password,
            sslmode="require"
        )
        return conn
    except Exception as e:
        st.warning(f"Lakebase unavailable: {str(e)}. Using session-only storage.")
        return None

# Persistent watchlist storage (Lakebase with session fallback)
def get_watchlist(user_email: str) -> pd.DataFrame:
    """Get watchlist from Lakebase, fallback to session state."""
    conn = get_lakebase_connection()
    
    if conn:
        try:
            cursor = conn.cursor()
            cursor.execute(
                """SELECT wt.ticker, wt.added_at 
                   FROM watchlist_tickers wt
                   JOIN watchlists w ON wt.watchlist_id = w.watchlist_id
                   JOIN users u ON w.user_id = u.user_id
                   WHERE u.email = %s AND w.name = 'default'
                   ORDER BY wt.added_at DESC""",
                (user_email,)
            )
            rows = cursor.fetchall()
            cursor.close()
            if rows:
                return pd.DataFrame(rows, columns=["ticker", "added_at"])
        except Exception:
            pass
    
    # Fallback: session state
    if "watchlist" not in st.session_state:
        st.session_state.watchlist = []
    if not st.session_state.watchlist:
        return pd.DataFrame(columns=["ticker", "added_at"])
    return pd.DataFrame(st.session_state.watchlist)

def add_to_watchlist(ticker: str, user_email: str):
    """Add ticker to Lakebase watchlist, fallback to session state."""
    conn = get_lakebase_connection()
    
    if conn:
        try:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO users (email, created_at) VALUES (%s, NOW()) ON CONFLICT (email) DO NOTHING",
                (user_email,)
            )
            cursor.execute("SELECT user_id FROM users WHERE email = %s", (user_email,))
            user_id = cursor.fetchone()[0]
            
            cursor.execute(
                "SELECT watchlist_id FROM watchlists WHERE user_id = %s AND name = 'default'",
                (user_id,)
            )
            result = cursor.fetchone()
            
            if result:
                watchlist_id = result[0]
            else:
                cursor.execute(
                    "INSERT INTO watchlists (user_id, name, created_at) VALUES (%s, 'default', NOW()) RETURNING watchlist_id",
                    (user_id,)
                )
                watchlist_id = cursor.fetchone()[0]
            
            cursor.execute(
                """INSERT INTO watchlist_tickers (watchlist_id, ticker, added_at) 
                   VALUES (%s, %s, NOW()) 
                   ON CONFLICT (watchlist_id, ticker) DO NOTHING""",
                (watchlist_id, ticker.upper())
            )
            conn.commit()
            cursor.close()
            return
        except Exception:
            pass
    
    # Fallback: session state
    if "watchlist" not in st.session_state:
        st.session_state.watchlist = []
    st.session_state.watchlist.append({"ticker": ticker.upper(), "added_at": datetime.now()})

def remove_from_watchlist(ticker: str, user_email: str):
    """Remove ticker from Lakebase watchlist, fallback to session state."""
    conn = get_lakebase_connection()
    
    if conn:
        try:
            cursor = conn.cursor()
            cursor.execute(
                """DELETE FROM watchlist_tickers 
                   WHERE watchlist_id IN (
                       SELECT w.watchlist_id 
                       FROM watchlists w 
                       JOIN users u ON w.user_id = u.user_id 
                       WHERE u.email = %s AND w.name = 'default'
                   ) AND ticker = %s""",
                (user_email, ticker.upper())
            )
            conn.commit()
            cursor.close()
            return
        except Exception:
            pass
    
    # Fallback: session state
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
    
    try:
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
            st.info(f"No news found for {ticker}. Run notebook 03 to generate embeddings.")
    except Exception:
        st.info("News search unavailable. Run notebook 03 to generate embeddings first.")
    
    conn = get_lakebase_connection()
    if conn:
        st.success("✅ Watchlist is persistently stored in Lakebase Postgres.")
    else:
        st.info("💡 Watchlist is temporarily stored in session (resets on refresh). Lakebase not available.")

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
    """Process chat message using the AI agent."""
    if not AGENT_AVAILABLE:
        return """⚠️ AI Agent is currently unavailable. 
        
Please ensure:
1. The agent module (04_Agent.py) is properly configured
2. All required dependencies are installed
3. Lakebase Postgres is accessible

You can still use:
- Sidebar watchlist (session-only)
- Stock price lookups  
- Manual search"""
    
    try:
        # Call the actual agent
        response = run_agent(message, st.session_state.user_email)
        return response
    
    except Exception as e:
        return f"""⚠️ Agent error: {str(e)}
        
Falling back to basic features. You can:
- Add tickers to watchlist via sidebar
- Ask for stock prices (e.g., "price of AAPL")
- Search news (if embeddings are available)"""

# Main app
def main():
    # Page configuration must be first Streamlit command
    st.set_page_config(
        page_title="AI Stock Market Research Assistant",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "user_email" not in st.session_state:
        st.session_state.user_email = "demo@example.com"
    
    st.title("🤖 AI Stock Market Research Assistant")
    
    # Check if embeddings are available
    index = load_search_index()
    if index is None or (isinstance(index, pd.DataFrame) and index.empty):
        st.info("ℹ️ Semantic search is unavailable. Run **notebook 03** to generate embeddings. Stock prices and watchlist work normally.")
    
    # Render sidebar
    render_sidebar()
    
    # Main content area
    if "selected_ticker" in st.session_state and st.session_state.selected_ticker:
        render_stock_details(st.session_state.selected_ticker)
    else:
        render_chat()

if __name__ == "__main__":
    main()
