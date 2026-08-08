"""04 - AI Stock Market Research Assistant Agent

This implements a LangGraph-based AI agent with tool-calling capabilities:

AGENT FEATURES:
- LangGraph state graph for agent workflow
- 11 tools for market data, semantic search, watchlist, and notes
- Reads from Yahoo Finance API (real-time data)
- Writes to Lakebase Postgres (persistent storage)
- Semantic search over embeddings (Vector Search)

TOOLS (Read):
- get_current_price: Current stock price and info
- get_historical_prices: Historical OHLCV data
- get_company_fundamentals: Company profile and metrics
- search_news_and_filings: Semantic search for news
- search_company_profiles: Semantic search for companies
- get_watchlist: User's saved tickers
- get_research_notes: Saved research notes
- compare_tickers: Multi-ticker comparison

TOOLS (Write):
- add_to_watchlist: Add ticker to watchlist
- remove_from_watchlist: Remove ticker
- save_research_note: Save analysis to database

This demonstrates:
- Agent reasoning and tool selection
- Read and write operations (Lakebase Postgres)
- Semantic retrieval (Vector Search)
- Third-party API integration (Yahoo Finance)
"""

from databricks.sdk import WorkspaceClient
from databricks.vector_search.client import VectorSearchClient
import psycopg2
import mlflow.deployments
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, TypedDict, Annotated
import pandas as pd
import yfinance as yf
import operator
import uuid
import time

# LangGraph imports
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from databricks_langchain import ChatDatabricks

# Configuration
LAKEBASE_PROJECT_NAME = "new_database"
LAKEBASE_BRANCH_NAME = "production"
LAKEBASE_DATABASE_NAME = "databricks_postgres"

CATALOG = "stock_research_capstone"
SCHEMA = "main"
VECTOR_SEARCH_ENDPOINT = "vector_search"
VECTOR_INDEX_NAME = f"{CATALOG}.{SCHEMA}.text_embeddings_index"
EMBEDDING_MODEL_ENDPOINT = "databricks-bge-large-en"

DEFAULT_USER_EMAIL = "test@example.com"

# Initialize clients
w = WorkspaceClient()
vsc = VectorSearchClient()
mlflow_client = mlflow.deployments.get_deploy_client("databricks")


def get_lakebase_connection():
    """Get psycopg2 connection to Lakebase Postgres."""
    endpoints = list(w.postgres.list_endpoints(
        parent=f"projects/{LAKEBASE_PROJECT_NAME}/branches/{LAKEBASE_BRANCH_NAME}"
    ))
    
    if not endpoints:
        raise ValueError("No Lakebase endpoints found")
    
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


def get_yahoo_finance_data(ticker: str) -> dict:
    """Fetch current data from Yahoo Finance."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return {"info": info, "error": None}
    except Exception as e:
        return {"info": {}, "error": str(e)}


def get_yahoo_historical_data(ticker: str, period: str = "1mo") -> pd.DataFrame:
    """Fetch historical price data."""
    try:
        stock = yf.Ticker(ticker)
        return stock.history(period=period)
    except Exception as e:
        return pd.DataFrame()


def get_embedding(text: str) -> List[float]:
    """Generate embedding for a text query."""
    response = mlflow_client.predict(
        endpoint=EMBEDDING_MODEL_ENDPOINT,
        inputs={"input": [text[:8000]]}
    )
    return response["data"][0]["embedding"]


def semantic_search(query: str, doc_type: Optional[str] = None, ticker: Optional[str] = None, num_results: int = 5):
    """Perform semantic search over embeddings."""
    query_emb = get_embedding(query)
    
    index = vsc.get_index(
        endpoint_name=VECTOR_SEARCH_ENDPOINT,
        index_name=VECTOR_INDEX_NAME
    )
    
    filters = []
    if doc_type:
        filters.append(f"doc_type = '{doc_type}'")
    if ticker:
        filters.append(f"ticker = '{ticker.upper()}'")
    
    filter_str = " AND ".join(filters) if filters else None
    
    results = index.similarity_search(
        query_vector=query_emb,
        columns=["doc_id", "doc_type", "ticker", "title", "text"],
        num_results=num_results,
        filters=filter_str
    )
    
    return results.get("result", {}).get("data_array", [])


# ============================================================================
# AGENT TOOLS - Market Data (Read)
# ============================================================================

@tool
def get_current_price(ticker: str) -> str:
    """Get the current price and basic info for a stock ticker.
    
    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL', 'TSLA')
    
    Returns:
        Current price, change, and volume information
    """
    data = get_yahoo_finance_data(ticker)
    
    if data["error"]:
        return f"Error fetching data for {ticker}: {data['error']}"
    
    info = data["info"]
    current_price = info.get('currentPrice', info.get('regularMarketPrice', 'N/A'))
    prev_close = info.get('previousClose', 0)
    change = current_price - prev_close if isinstance(current_price, (int, float)) and prev_close else 0
    change_pct = (change / prev_close * 100) if prev_close else 0
    
    return f"""
{info.get('longName', ticker)} ({ticker})
Current Price: ${current_price:.2f}
Change: ${change:+.2f} ({change_pct:+.2f}%)
Volume: {info.get('volume', 0):,}
Market Cap: ${info.get('marketCap', 0):,}
"""


@tool
def get_historical_prices(ticker: str, days: int = 30) -> str:
    """Get historical price data for a ticker.
    
    Args:
        ticker: Stock ticker symbol
        days: Number of days of history (default: 30)
    
    Returns:
        Summary of historical price performance
    """
    period = "1mo" if days <= 30 else "3mo" if days <= 90 else "6mo"
    hist = get_yahoo_historical_data(ticker, period=period)
    
    if hist.empty:
        return f"No historical data available for {ticker}"
    
    first_price = hist['Close'].iloc[0]
    last_price = hist['Close'].iloc[-1]
    high = hist['High'].max()
    low = hist['Low'].min()
    change = ((last_price - first_price) / first_price) * 100
    
    return f"""
{ticker} - {days}-Day Historical Performance:
  Starting Price: ${first_price:.2f}
  Ending Price: ${last_price:.2f}
  Change: {change:+.2f}%
  High: ${high:.2f}
  Low: ${low:.2f}
  Avg Volume: {hist['Volume'].mean():,.0f}
"""


@tool
def get_company_fundamentals(ticker: str) -> str:
    """Get company fundamentals and profile information.
    
    Args:
        ticker: Stock ticker symbol
    
    Returns:
        Company profile, sector, industry, and key metrics
    """
    data = get_yahoo_finance_data(ticker)
    
    if data["error"]:
        return f"Error fetching data for {ticker}: {data['error']}"
    
    info = data["info"]
    
    return f"""
{info.get('longName', ticker)} ({ticker})

Sector: {info.get('sector', 'N/A')}
Industry: {info.get('industry', 'N/A')}
Employees: {info.get('fullTimeEmployees', 'N/A'):,}

Market Cap: ${info.get('marketCap', 0):,}
P/E Ratio: {info.get('trailingPE', 'N/A')}
Forward P/E: {info.get('forwardPE', 'N/A')}
Dividend Yield: {info.get('dividendYield', 0)*100:.2f}%

52-Week High: ${info.get('fiftyTwoWeekHigh', 'N/A')}
52-Week Low: ${info.get('fiftyTwoWeekLow', 'N/A')}

Description:
{info.get('longBusinessSummary', 'No description available')[:500]}
"""


# ============================================================================
# AGENT TOOLS - Semantic Search (Read)
# ============================================================================

@tool
def search_news_and_filings(query: str, ticker: Optional[str] = None) -> str:
    """Search for relevant news articles using semantic search.
    
    Args:
        query: Natural language query (e.g., "interest rate exposure")
        ticker: Optional ticker to filter results
    
    Returns:
        Relevant news articles and their content
    """
    results = semantic_search(query, doc_type="news", ticker=ticker, num_results=5)
    
    if not results:
        return f"No news found for query: {query}"
    
    output = f"News Results for '{query}':\\n\\n"
    for i, row in enumerate(results, 1):
        doc_id, doc_type, tkr, title, text, *_ = row
        output += f"{i}. [{tkr}] {title}\\n"
        output += f"   {text[:200]}...\\n\\n"
    
    return output


@tool
def search_company_profiles(query: str) -> str:
    """Search for companies matching a description using semantic search.
    
    Args:
        query: Natural language description (e.g., "EV manufacturers")
    
    Returns:
        Matching company profiles
    """
    results = semantic_search(query, doc_type="company_profile", num_results=5)
    
    if not results:
        return f"No companies found for query: {query}"
    
    output = f"Company Profiles for '{query}':\\n\\n"
    for i, row in enumerate(results, 1):
        doc_id, doc_type, tkr, title, text, *_ = row
        output += f"{i}. [{tkr}] {title}\\n"
        output += f"   {text[:300]}...\\n\\n"
    
    return output


# ============================================================================
# AGENT TOOLS - Watchlist Management (Read + Write)
# ============================================================================

@tool
def add_to_watchlist(ticker: str, user_email: str = DEFAULT_USER_EMAIL, watchlist_name: str = "default") -> str:
    """Add a ticker to a user's watchlist (WRITE operation to Lakebase).
    
    Args:
        ticker: Stock ticker symbol
        user_email: User's email
        watchlist_name: Name of the watchlist
    
    Returns:
        Confirmation message
    """
    try:
        conn = get_lakebase_connection()
        cursor = conn.cursor()
        
        # Ensure user exists
        cursor.execute(
            "INSERT INTO users (email, created_at) VALUES (%s, NOW()) ON CONFLICT (email) DO NOTHING",
            (user_email,)
        )
        
        # Get user_id
        cursor.execute("SELECT user_id FROM users WHERE email = %s", (user_email,))
        user_id = cursor.fetchone()[0]
        
        # Ensure watchlist exists - check first since table has no unique constraint on (user_id, name)
        cursor.execute(
            "SELECT watchlist_id FROM watchlists WHERE user_id = %s AND name = %s",
            (user_id, watchlist_name)
        )
        result = cursor.fetchone()
        
        if result:
            watchlist_id = result[0]
        else:
            cursor.execute(
                """INSERT INTO watchlists (user_id, name, created_at) 
                   VALUES (%s, %s, NOW()) 
                   RETURNING watchlist_id""",
                (user_id, watchlist_name)
            )
            watchlist_id = cursor.fetchone()[0]
        
        # Add ticker
        cursor.execute(
            """INSERT INTO watchlist_tickers (watchlist_id, ticker, added_at) 
               VALUES (%s, %s, NOW()) 
               ON CONFLICT (watchlist_id, ticker) DO NOTHING""",
            (watchlist_id, ticker.upper())
        )
        
        conn.commit()
        cursor.close()
        conn.close()
        
        return f"✓ Added {ticker.upper()} to watchlist '{watchlist_name}'"
    
    except Exception as e:
        return f"Error adding to watchlist: {str(e)}"


@tool
def remove_from_watchlist(ticker: str, user_email: str = DEFAULT_USER_EMAIL, watchlist_name: str = "default") -> str:
    """Remove a ticker from watchlist (WRITE operation to Lakebase).
    
    Args:
        ticker: Stock ticker symbol
        user_email: User's email
        watchlist_name: Name of the watchlist
    
    Returns:
        Confirmation message
    """
    try:
        conn = get_lakebase_connection()
        cursor = conn.cursor()
        
        cursor.execute(
            """DELETE FROM watchlist_tickers 
               WHERE watchlist_id IN (
                   SELECT w.watchlist_id 
                   FROM watchlists w 
                   JOIN users u ON w.user_id = u.user_id 
                   WHERE u.email = %s AND w.name = %s
               ) AND ticker = %s""",
            (user_email, watchlist_name, ticker.upper())
        )
        
        conn.commit()
        rows_deleted = cursor.rowcount
        cursor.close()
        conn.close()
        
        if rows_deleted > 0:
            return f"✓ Removed {ticker.upper()} from watchlist '{watchlist_name}'"
        else:
            return f"{ticker.upper()} not found in watchlist '{watchlist_name}'"
    
    except Exception as e:
        return f"Error removing from watchlist: {str(e)}"


@tool
def get_watchlist(user_email: str = DEFAULT_USER_EMAIL, watchlist_name: str = "default") -> str:
    """Get user's watchlist (READ operation from Lakebase).
    
    Args:
        user_email: User's email
        watchlist_name: Name of the watchlist
    
    Returns:
        List of tickers in the watchlist
    """
    try:
        conn = get_lakebase_connection()
        cursor = conn.cursor()
        
        cursor.execute(
            """SELECT wt.ticker, wt.added_at 
               FROM watchlist_tickers wt
               JOIN watchlists w ON wt.watchlist_id = w.watchlist_id
               JOIN users u ON w.user_id = u.user_id
               WHERE u.email = %s AND w.name = %s
               ORDER BY wt.added_at DESC""",
            (user_email, watchlist_name)
        )
        
        tickers = cursor.fetchall()
        cursor.close()
        conn.close()
        
        if not tickers:
            return f"Watchlist '{watchlist_name}' is empty"
        
        output = f"Watchlist '{watchlist_name}' ({len(tickers)} tickers):\\n\\n"
        for ticker, added_at in tickers:
            output += f"  • {ticker} (added {added_at})\\n"
        
        return output
    
    except Exception as e:
        return f"Error fetching watchlist: {str(e)}"


# ============================================================================
# AGENT TOOLS - Research Notes (Read + Write)
# ============================================================================

@tool
def save_research_note(ticker: str, note: str, user_email: str = DEFAULT_USER_EMAIL) -> str:
    """Save a research note (WRITE operation to Lakebase).
    
    Args:
        ticker: Stock ticker symbol
        note: The research note content
        user_email: User's email
    
    Returns:
        Confirmation with note ID
    """
    try:
        conn = get_lakebase_connection()
        cursor = conn.cursor()
        
        # Ensure user exists and get user_id
        cursor.execute(
            "INSERT INTO users (email, created_at) VALUES (%s, NOW()) ON CONFLICT (email) DO NOTHING",
            (user_email,)
        )
        cursor.execute("SELECT user_id FROM users WHERE email = %s", (user_email,))
        user_id = cursor.fetchone()[0]
        
        # Insert note with correct schema: user_id (not user_email), note_id (not id)
        cursor.execute(
            """INSERT INTO research_notes (user_id, ticker, content, created_at) 
               VALUES (%s, %s, %s, NOW()) 
               RETURNING note_id""",
            (user_id, ticker.upper(), note)
        )
        
        note_id = cursor.fetchone()[0]
        conn.commit()
        cursor.close()
        conn.close()
        
        return f"✓ Research note saved (ID: {note_id}) for {ticker.upper()}"
    
    except Exception as e:
        return f"Error saving note: {str(e)}"


@tool
def get_research_notes(ticker: Optional[str] = None, user_email: str = DEFAULT_USER_EMAIL, limit: int = 10) -> str:
    """Get research notes (READ operation from Lakebase).
    
    Args:
        ticker: Optional ticker to filter notes
        user_email: User's email
        limit: Maximum number of notes
    
    Returns:
        List of research notes
    """
    try:
        conn = get_lakebase_connection()
        cursor = conn.cursor()
        
        if ticker:
            cursor.execute(
                """SELECT id, ticker, content, created_at 
                   FROM research_notes 
                   WHERE user_email = %s AND ticker = %s
                   ORDER BY created_at DESC 
                   LIMIT %s""",
                (user_email, ticker.upper(), limit)
            )
        else:
            cursor.execute(
                """SELECT id, ticker, content, created_at 
                   FROM research_notes 
                   WHERE user_email = %s
                   ORDER BY created_at DESC 
                   LIMIT %s""",
                (user_email, limit)
            )
        
        notes = cursor.fetchall()
        cursor.close()
        conn.close()
        
        if not notes:
            return "No research notes found"
        
        output = "Research Notes:\\n\\n"
        for note_id, tkr, content, created_at in notes:
            output += f"[{note_id}] {tkr} - {created_at}\\n"
            output += f"{content[:200]}...\\n\\n"
        
        return output
    
    except Exception as e:
        return f"Error fetching notes: {str(e)}"


# ============================================================================
# AGENT TOOLS - Comparison
# ============================================================================

@tool
def compare_tickers(tickers: List[str], days: int = 30) -> str:
    """Compare multiple tickers on performance.
    
    Args:
        tickers: List of ticker symbols
        days: Number of days for comparison
    
    Returns:
        Comparative analysis
    """
    if len(tickers) < 2:
        return "Please provide at least 2 tickers"
    
    comparisons = []
    period = "1mo" if days <= 30 else "3mo"
    
    for ticker in tickers:
        data = get_yahoo_finance_data(ticker)
        if data["error"]:
            continue
        
        info = data["info"]
        hist = get_yahoo_historical_data(ticker, period=period)
        
        if not hist.empty:
            first_price = hist['Close'].iloc[0]
            last_price = hist['Close'].iloc[-1]
            change = ((last_price - first_price) / first_price) * 100
            
            comparisons.append({
                'ticker': ticker,
                'price': last_price,
                'change_pct': change,
                'market_cap': info.get('marketCap', 0)
            })
    
    if not comparisons:
        return "Unable to fetch data for comparison"
    
    comparisons.sort(key=lambda x: x['change_pct'], reverse=True)
    
    output = f"Comparison ({days}-day performance):\\n\\n"
    for i, comp in enumerate(comparisons, 1):
        output += f"{i}. {comp['ticker']}: ${comp['price']:.2f} ({comp['change_pct']:+.2f}%)\\n"
    
    return output


# ============================================================================
# AGENT STATE AND GRAPH
# ============================================================================

class AgentState(TypedDict):
    messages: Annotated[List, operator.add]


# Collect all tools
tools = [
    get_current_price,
    get_historical_prices,
    get_company_fundamentals,
    search_news_and_filings,
    search_company_profiles,
    add_to_watchlist,
    remove_from_watchlist,
    get_watchlist,
    save_research_note,
    get_research_notes,
    compare_tickers
]


def should_continue(state: AgentState):
    """Decide whether to continue or end."""
    last_message = state["messages"][-1]
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tools"
    return "end"


# Initialize LLM with tool binding
llm = ChatDatabricks(
    endpoint="databricks-meta-llama-3-1-70b-instruct",
    temperature=0.1
)
llm_with_tools = llm.bind_tools(tools)

system_prompt = """You are an AI stock market research assistant with access to real-time market data, 
semantic search over news/filings, and persistent storage.

Your capabilities:
- Get current prices and historical data (Yahoo Finance)
- Search news and company profiles (Vector Search)
- Manage user watchlists (Lakebase Postgres)
- Save and retrieve research notes
- Compare multiple stocks

Always use tools to provide accurate, data-driven responses. When users ask about stocks, 
fetch real data rather than making assumptions."""

def call_model(state: AgentState):
    """Call the LLM to decide next action."""
    messages = state["messages"]
    
    # Add system prompt if this is the first message
    if len(messages) == 1 or not any(isinstance(m, SystemMessage) for m in messages):
        messages = [SystemMessage(content=system_prompt)] + messages
    
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


# Create the graph
workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", ToolNode(tools))
workflow.set_entry_point("agent")
workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
workflow.add_edge("tools", "agent")

agent_executor = workflow.compile()

print(f"✓ Agent configured with {len(tools)} tools")
print("  Read tools: 8 (market data, search, watchlist, notes)")
print("  Write tools: 3 (add/remove watchlist, save notes)")
print("\\nAgent ready!")


def run_agent(user_message: str, user_email: str = DEFAULT_USER_EMAIL) -> str:
    """Run the agent with a user message and return the final response.
    
    Args:
        user_message: User's input message
        user_email: User's email for personalized operations
    
    Returns:
        Agent's final response text
    """
    # Override default user email for tool calls
    global DEFAULT_USER_EMAIL
    original_default = DEFAULT_USER_EMAIL
    DEFAULT_USER_EMAIL = user_email
    
    try:
        # Create initial state with user message
        initial_state = {
            "messages": [HumanMessage(content=user_message)]
        }
        
        # Run the agent
        result = agent_executor.invoke(initial_state)
        
        # Extract final response
        final_messages = result["messages"]
        for msg in reversed(final_messages):
            if isinstance(msg, AIMessage) and msg.content:
                return msg.content
        
        return "I apologize, but I couldn't process your request."
    
    except Exception as e:
        return f"Error: {str(e)}"
    
    finally:
        # Restore default
        DEFAULT_USER_EMAIL = original_default


if __name__ == "__main__":
    print("\\n" + "="*80)
    print("AI STOCK MARKET RESEARCH ASSISTANT")
    print("="*80)
    print("\\nExample agent usage:")
    print("  run_agent('What is the current price of AAPL?')")
    print("  run_agent('Add TSLA to my watchlist')")
    print("  run_agent('Search for news about EV sector')")
    print("  run_agent('Compare AAPL, MSFT, and GOOGL')")
