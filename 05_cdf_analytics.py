# Databricks notebook source
# MAGIC %md
# MAGIC # CDF Analytics Pipeline
# MAGIC 
# MAGIC This notebook reads Change Data Feed (CDF) from Delta tables to create analytics:
# MAGIC - Agent tool call logs with CDF enabled
# MAGIC - Watchlist mutations tracking
# MAGIC - Daily aggregations for dashboard visualization

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, count, sum as _sum, date_trunc, current_timestamp,
    to_date, hour, window, lit
)
from delta.tables import DeltaTable
from datetime import datetime, timedelta

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

CATALOG = "stock_research_capstone"
SCHEMA = "main"

# Source tables with CDF enabled
TOOL_CALLS_TABLE = f"{CATALOG}.{SCHEMA}.agent_tool_calls"
WATCHLIST_LOG_TABLE = f"{CATALOG}.{SCHEMA}.watchlist_mutations"

# Analytics output tables
ANALYTICS_TABLE = f"{CATALOG}.{SCHEMA}.usage_analytics"
DAILY_STATS_TABLE = f"{CATALOG}.{SCHEMA}.daily_usage_stats"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Create Tool Calls Log Table (with CDF)

# COMMAND ----------

def create_tool_calls_table():
    """
    Create agent_tool_calls table to log every tool invocation.
    CDF enabled to track all changes.
    """
    spark.sql(f"""
        CREATE TABLE IF NOT EXISTS {TOOL_CALLS_TABLE} (
            call_id STRING,
            timestamp TIMESTAMP,
            user_id STRING,
            tool_name STRING,
            ticker STRING,
            success BOOLEAN,
            error_message STRING,
            execution_time_ms LONG
        )
        USING DELTA
        TBLPROPERTIES (
            'delta.enableChangeDataFeed' = 'true',
            'delta.columnMapping.mode' = 'name'
        )
    """)
    print(f"✅ Created {TOOL_CALLS_TABLE} with CDF enabled")

create_tool_calls_table()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Create Watchlist Mutations Log Table (with CDF)

# COMMAND ----------

def create_watchlist_log_table():
    """
    Create watchlist_mutations table to track add/remove operations.
    CDF enabled to track all watchlist changes.
    """
    spark.sql(f"""
        CREATE TABLE IF NOT EXISTS {WATCHLIST_LOG_TABLE} (
            mutation_id STRING,
            timestamp TIMESTAMP,
            user_id STRING,
            ticker STRING,
            operation STRING,
            source STRING
        )
        USING DELTA
        TBLPROPERTIES (
            'delta.enableChangeDataFeed' = 'true',
            'delta.columnMapping.mode' = 'name'
        )
    """)
    print(f"✅ Created {WATCHLIST_LOG_TABLE} with CDF enabled")

create_watchlist_log_table()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Read CDF and Create Analytics

# COMMAND ----------

def process_tool_calls_cdf(start_timestamp=None):
    """
    Read CDF from tool calls table and aggregate metrics.
    """
    if start_timestamp is None:
        start_timestamp = (datetime.now() - timedelta(days=7)).isoformat()
    
    cdf_df = spark.read.format("delta") \
        .option("readChangeFeed", "true") \
        .option("startingTimestamp", start_timestamp) \
        .table(TOOL_CALLS_TABLE)
    
    hourly_stats = cdf_df.filter(col("_change_type").isin(["insert", "update_postimage"])) \
        .withColumn("hour", date_trunc("hour", col("timestamp"))) \
        .groupBy("hour", "tool_name", "user_id") \
        .agg(
            count("*").alias("call_count"),
            _sum(col("success").cast("int")).alias("success_count"),
            count(col("error_message")).alias("error_count"),
            (count("*") - count(col("error_message"))).alias("successful_calls")
        )
    
    return hourly_stats

# COMMAND ----------

def process_watchlist_cdf(start_timestamp=None):
    """
    Read CDF from watchlist mutations and aggregate changes.
    """
    if start_timestamp is None:
        start_timestamp = (datetime.now() - timedelta(days=7)).isoformat()
    
    cdf_df = spark.read.format("delta") \
        .option("readChangeFeed", "true") \
        .option("startingTimestamp", start_timestamp) \
        .table(WATCHLIST_LOG_TABLE)
    
    daily_mutations = cdf_df.filter(col("_change_type").isin(["insert"])) \
        .withColumn("date", to_date(col("timestamp"))) \
        .groupBy("date", "operation", "user_id") \
        .agg(
            count("*").alias("mutation_count"),
            count(col("ticker").distinct()).alias("unique_tickers")
        )
    
    return daily_mutations

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Create Daily Analytics Table

# COMMAND ----------

def create_daily_stats_table():
    """
    Aggregate daily usage statistics from CDF data.
    """
    spark.sql(f"""
        CREATE TABLE IF NOT EXISTS {DAILY_STATS_TABLE} (
            date DATE,
            user_id STRING,
            total_tool_calls INT,
            successful_calls INT,
            failed_calls INT,
            watchlist_additions INT,
            watchlist_removals INT,
            unique_tickers_viewed INT,
            most_used_tool STRING,
            last_updated TIMESTAMP
        )
        USING DELTA
        PARTITIONED BY (date)
    """)
    print(f"✅ Created {DAILY_STATS_TABLE}")

create_daily_stats_table()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: ETL Job - Incremental CDF Processing

# COMMAND ----------

def run_cdf_etl():
    """
    Main ETL job to process CDF and update analytics tables.
    Run this on a schedule (e.g., hourly or daily).
    """
    print(f"🔄 Starting CDF ETL at {datetime.now()}")
    
    try:
        last_run = spark.sql(f"""
            SELECT MAX(last_updated) as max_timestamp 
            FROM {DAILY_STATS_TABLE}
        """).collect()[0]["max_timestamp"]
        
        if last_run:
            start_time = last_run.isoformat()
        else:
            start_time = (datetime.now() - timedelta(days=30)).isoformat()
    except:
        start_time = (datetime.now() - timedelta(days=30)).isoformat()
    
    print(f"📅 Processing CDF from {start_time}")
    
    try:
        tool_stats = process_tool_calls_cdf(start_time)
        tool_stats.write.format("delta").mode("append").saveAsTable(f"{CATALOG}.{SCHEMA}.hourly_tool_stats")
        print(f"✅ Processed tool calls CDF: {tool_stats.count()} records")
    except Exception as e:
        print(f"⚠️  Tool calls CDF: {str(e)}")
    
    try:
        watchlist_stats = process_watchlist_cdf(start_time)
        watchlist_stats.write.format("delta").mode("append").saveAsTable(f"{CATALOG}.{SCHEMA}.daily_watchlist_stats")
        print(f"✅ Processed watchlist CDF: {watchlist_stats.count()} records")
    except Exception as e:
        print(f"⚠️  Watchlist CDF: {str(e)}")
    
    print(f"✅ CDF ETL completed at {datetime.now()}")

run_cdf_etl()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6: Query Analytics for Dashboard

# COMMAND ----------

def get_usage_stats_last_7_days():
    """
    Get usage statistics for the last 7 days for dashboard display.
    """
    return spark.sql(f"""
        SELECT 
            date,
            SUM(total_tool_calls) as total_calls,
            SUM(successful_calls) as successful,
            SUM(failed_calls) as failed
        FROM {DAILY_STATS_TABLE}
        WHERE date >= current_date() - INTERVAL 7 DAYS
        GROUP BY date
        ORDER BY date DESC
    """)

# COMMAND ----------

import uuid

def insert_sample_tool_calls():
    """Insert sample tool call logs for testing."""
    sample_data = [
        (str(uuid.uuid4()), datetime.now(), "user123", "get_stock_price", "AAPL", True, None, 150),
        (str(uuid.uuid4()), datetime.now(), "user123", "search_by_sector", "tech", True, None, 320),
        (str(uuid.uuid4()), datetime.now(), "user123", "add_to_watchlist", "TSLA", True, None, 89),
    ]
    
    df = spark.createDataFrame(sample_data, ["call_id", "timestamp", "user_id", "tool_name", "ticker", "success", "error_message", "execution_time_ms"])
    df.write.format("delta").mode("append").saveAsTable(TOOL_CALLS_TABLE)
    print(f"✅ Inserted {df.count()} sample records")

insert_sample_tool_calls()

print("\n✅ CDF Analytics Pipeline Complete!")