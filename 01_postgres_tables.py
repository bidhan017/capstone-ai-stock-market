"""
01 - Lakebase Schema Setup

Create relational Postgres tables in Lakebase for the stock research assistant
"""

# SQL queries to create tables
SQL_COMMANDS = """
-- Users
CREATE TABLE IF NOT EXISTS users (
    user_id       SERIAL PRIMARY KEY,
    email         TEXT UNIQUE NOT NULL,
    display_name  TEXT,
    created_at    TIMESTAMPTZ DEFAULT now()
);

-- Watchlists
CREATE TABLE IF NOT EXISTS watchlists (
    watchlist_id  SERIAL PRIMARY KEY,
    user_id       INT REFERENCES users(user_id),
    name          TEXT NOT NULL,
    description   TEXT,
    created_at    TIMESTAMPTZ DEFAULT now()
);

-- Watchlist tickers (many-to-many)
CREATE TABLE IF NOT EXISTS watchlist_tickers (
    watchlist_id  INT REFERENCES watchlists(watchlist_id),
    ticker        TEXT NOT NULL,
    added_at      TIMESTAMPTZ DEFAULT now(),
    PRIMARY KEY (watchlist_id, ticker)
);

-- Companies
CREATE TABLE IF NOT EXISTS companies (
    ticker        TEXT PRIMARY KEY,
    name          TEXT NOT NULL,
    sector        TEXT,
    industry      TEXT,
    market_cap    BIGINT,
    description   TEXT,
    updated_at    TIMESTAMPTZ DEFAULT now()
);

-- Price snapshots
CREATE TABLE IF NOT EXISTS price_snapshots (
    ticker        TEXT NOT NULL,
    snapshot_date DATE NOT NULL,
    open_price    NUMERIC(12,4),
    close_price   NUMERIC(12,4),
    high_price    NUMERIC(12,4),
    low_price     NUMERIC(12,4),
    volume        BIGINT,
    PRIMARY KEY (ticker, snapshot_date)
);

-- News articles
CREATE TABLE IF NOT EXISTS news_articles (
    article_id    SERIAL PRIMARY KEY,
    ticker        TEXT,
    title         TEXT NOT NULL,
    source        TEXT,
    published_at  TIMESTAMPTZ,
    url           TEXT,
    summary       TEXT,
    full_text     TEXT,
    ingested_at   TIMESTAMPTZ DEFAULT now()
);

-- Research notes
CREATE TABLE IF NOT EXISTS research_notes (
    note_id       SERIAL PRIMARY KEY,
    user_id       INT REFERENCES users(user_id),
    ticker        TEXT,
    title         TEXT,
    content       TEXT NOT NULL,
    created_at    TIMESTAMPTZ DEFAULT now()
);

-- Analysis reports
CREATE TABLE IF NOT EXISTS analysis_reports (
    report_id     SERIAL PRIMARY KEY,
    user_id       INT REFERENCES users(user_id),
    tickers       TEXT[],
    thesis        TEXT,
    report_body   TEXT NOT NULL,
    created_at    TIMESTAMPTZ DEFAULT now()
);
"""

if __name__ == "__main__":
    print("To execute these SQL commands, run them against your Lakebase Postgres database")
    print("You can use psql, DBeaver, or execute via Python with psycopg2")
    print("\nSQL Commands:")
    print(SQL_COMMANDS)
