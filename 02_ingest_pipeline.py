"""02 - Ingest Pipeline: Bronze to Silver Data Transformation

This pipeline:
1. Fetches stock data from Yahoo Finance API (yfinance)
2. Writes raw data to Bronze tables (Unity Catalog)
3. Transforms and cleans data into Silver tables
4. Optionally syncs to Lakebase Postgres for low-latency reads

Data sources:
- Price history: Daily OHLCV data
- Company profiles: Fundamentals and descriptions
- News articles: Recent headlines and content
"""

import yfinance as yf
from datetime import date, timedelta
from pyspark.sql import functions as F, SparkSession

# Configuration
CATALOG = 'stock_research_capstone'
SCHEMA = 'main'
TICKERS = ['AAPL', 'AMZN', 'NFLX', 'GOOGL', 'META', 'TSLA', 'NVDA', 'MSFT']

# Initialize Spark (if running outside Databricks notebooks)
try:
    spark = SparkSession.builder.getOrCreate()
except:
    print("Warning: SparkSession not available")


def fetch_price_history(ticker: str, days_back: int = 365) -> list:
    """Fetch daily OHLCV data for a ticker from Yahoo Finance.
    
    Args:
        ticker: Stock symbol (e.g., 'AAPL')
        days_back: Number of days of history to fetch
        
    Returns:
        List of dictionaries with price data
    """
    end_date = date.today()
    start_date = end_date - timedelta(days=days_back)
    
    stock = yf.Ticker(ticker)
    hist = stock.history(start=start_date, end=end_date, interval="1d")
    
    records = []
    for idx, row in hist.iterrows():
        records.append({
            "date": idx.strftime("%Y-%m-%d"),
            "open": float(row['Open']),
            "high": float(row['High']),
            "low": float(row['Low']),
            "close": float(row['Close']),
            "volume": int(row['Volume'])
        })
    return records


def fetch_company_profile(ticker: str) -> dict:
    """Fetch company fundamentals/profile from Yahoo Finance.
    
    Args:
        ticker: Stock symbol
        
    Returns:
        Dictionary with company information
    """
    stock = yf.Ticker(ticker)
    info = stock.info
    
    return {
        "name": info.get('longName', ticker),
        "sector": info.get('sector', 'Unknown'),
        "industry": info.get('industry', 'Unknown'),
        "marketCap": info.get('marketCap', 0),
        "description": info.get('longBusinessSummary', ''),
        "website": info.get('website', ''),
        "employees": info.get('fullTimeEmployees', 0),
        "country": info.get('country', ''),
        "currency": info.get('currency', 'USD')
    }


def fetch_news(ticker: str, limit: int = 20) -> list:
    """Fetch recent news articles for a ticker from Yahoo Finance.
    
    Args:
        ticker: Stock symbol
        limit: Maximum number of articles
        
    Returns:
        List of news article dictionaries
    """
    stock = yf.Ticker(ticker)
    news = stock.news[:limit] if hasattr(stock, 'news') else []
    
    articles = []
    for item in news:
        articles.append({
            "title": item.get('title', ''),
            "source": item.get('publisher', 'Yahoo Finance'),
            "publishedAt": item.get('providerPublishTime', 0),
            "url": item.get('link', ''),
            "summary": item.get('title', ''),
            "content": item.get('title', '')
        })
    return articles


def ingest_bronze_prices(spark, tickers: list):
    """Fetch and write raw price data to Bronze table."""
    all_prices = []
    for ticker in tickers:
        try:
            prices = fetch_price_history(ticker, days_back=365)
            for p in prices:
                p["ticker"] = ticker
            all_prices.extend(prices)
            print(f" ✓ {ticker}: {len(prices)} records")
        except Exception as e:
            print(f" X {ticker}: {e}")
    
    if all_prices:
        df_bronze_prices = spark.createDataFrame(all_prices)
        df_bronze_prices.write.mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.bronze_prices")
        print(f"\nBronze prices: {len(all_prices)} rows written")
        return df_bronze_prices
    return None


def ingest_bronze_companies(spark, tickers: list):
    """Fetch and write company profiles to Bronze table."""
    all_companies = []
    for ticker in tickers:
        try:
            profile = fetch_company_profile(ticker)
            profile["ticker"] = ticker
            all_companies.append(profile)
            print(f" ✓ {ticker}")
        except Exception as e:
            print(f" X {ticker}: {e}")
    
    if all_companies:
        df_bronze_companies = spark.createDataFrame(all_companies)
        df_bronze_companies.write.mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.bronze_companies")
        print(f"\nBronze companies: {len(all_companies)} rows written")
        return df_bronze_companies
    return None


def ingest_bronze_news(spark, tickers: list):
    """Fetch and write news articles to Bronze table."""
    all_news = []
    for ticker in tickers:
        try:
            articles = fetch_news(ticker, limit=50)
            for a in articles:
                a["ticker"] = ticker
            all_news.extend(articles)
            print(f" ✓ {ticker}: {len(articles)} articles")
        except Exception as e:
            print(f" X {ticker}: {e}")
    
    if all_news:
        df_bronze_news = spark.createDataFrame(all_news)
        df_bronze_news.write.mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.bronze_news")
        print(f"\nBronze news: {len(all_news)} articles written")
        return df_bronze_news
    return None


def transform_silver_prices(spark):
    """Transform Bronze prices to cleaned Silver table."""
    df_silver_prices = (
        spark.table(f"{CATALOG}.{SCHEMA}.bronze_prices")
        .select(
            F.col("ticker"),
            F.to_date(F.col("date")).alias("snapshot_date"),
            F.col("open").cast("decimal(12,4)").alias("open_price"),
            F.col("close").cast("decimal(12,4)").alias("close_price"),
            F.col("high").cast("decimal(12,4)").alias("high_price"),
            F.col("low").cast("decimal(12,4)").alias("low_price"),
            F.col("volume").cast("bigint")
        )
        .dropDuplicates(["ticker", "snapshot_date"])
        .filter(F.col("snapshot_date").isNotNull())
    )
    
    df_silver_prices.write.mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.silver_prices")
    print(f"Silver prices: {df_silver_prices.count()} rows")
    return df_silver_prices


def transform_silver_companies(spark):
    """Transform Bronze companies to cleaned Silver table."""
    df_silver_companies = (
        spark.table(f"{CATALOG}.{SCHEMA}.bronze_companies")
        .select(
            F.col("ticker"),
            F.col("name").alias("company_name"),
            F.col("sector"),
            F.col("industry"),
            F.col("marketCap").cast("bigint").alias("market_cap"),
            F.col("description").alias("company_description"),
            F.current_timestamp().alias("updated_at")
        )
        .dropDuplicates(["ticker"])
    )
    
    df_silver_companies.write.mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.silver_companies")
    print(f"Silver companies: {df_silver_companies.count()} rows")
    return df_silver_companies


def transform_silver_news(spark):
    """Transform Bronze news to cleaned Silver table."""
    df_silver_news = (
        spark.table(f"{CATALOG}.{SCHEMA}.bronze_news")
        .select(
            F.col("ticker"),
            F.col("title"),
            F.col("source"),
            F.to_timestamp(F.col("publishedAt")).alias("published_at"),
            F.col("url"),
            F.col("summary"),
            F.col("content").alias("full_text"),
            F.current_timestamp().alias("ingested_at")
        )
        .filter(F.col("title").isNotNull())
    )
    
    df_silver_news.write.mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.silver_news")
    print(f"Silver news: {df_silver_news.count()} rows")
    return df_silver_news


def run_pipeline():
    """Execute the complete ETL pipeline."""
    print("=== Stock Market Data Ingestion Pipeline ===")
    print(f"Target: {CATALOG}.{SCHEMA}")
    print(f"Tickers: {', '.join(TICKERS)}\n")
    
    # Create catalog and schema
    spark.sql(f"CREATE CATALOG IF NOT EXISTS {CATALOG}")
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
    
    # Bronze ingestion
    print("\n[1/6] Ingesting Bronze: Price History")
    ingest_bronze_prices(spark, TICKERS)
    
    print("\n[2/6] Ingesting Bronze: Company Profiles")
    ingest_bronze_companies(spark, TICKERS)
    
    print("\n[3/6] Ingesting Bronze: News Articles")
    ingest_bronze_news(spark, TICKERS)
    
    # Silver transformation
    print("\n[4/6] Transforming Silver: Prices")
    transform_silver_prices(spark)
    
    print("\n[5/6] Transforming Silver: Companies")
    transform_silver_companies(spark)
    
    print("\n[6/6] Transforming Silver: News")
    transform_silver_news(spark)
    
    print("\n✅ Pipeline complete!")
    print(f"\nData available at:")
    print(f"  - {CATALOG}.{SCHEMA}.silver_prices")
    print(f"  - {CATALOG}.{SCHEMA}.silver_companies")
    print(f"  - {CATALOG}.{SCHEMA}.silver_news")


if __name__ == "__main__":
    run_pipeline()
