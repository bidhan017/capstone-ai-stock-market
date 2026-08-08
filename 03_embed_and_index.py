"""03 - Embeddings and Vector Search Index

This script demonstrates DISTRIBUTED SPARK PROCESSING for generating embeddings:
1. Reads news articles and company profiles from Unity Catalog Silver tables
2. Uses Pandas UDF to generate embeddings in parallel across Spark workers
3. Writes embeddings to a Delta table with idempotent overwrite
4. Creates a Vector Search index for semantic retrieval
5. Tests semantic search functionality

KEY SPARK FEATURES:
- Pandas UDF for distributed embedding generation
- Spark DataFrame operations for data transformation
- Delta table writes with overwrite mode
- Unity Catalog integration
"""

from pyspark.sql import functions as F, SparkSession
from pyspark.sql.types import *
import mlflow.deployments
from databricks.vector_search.client import VectorSearchClient
import pandas as pd
import time

# Configuration
CATALOG = 'stock_research_capstone'
SCHEMA = 'main'
EMBEDDING_MODEL_ENDPOINT = 'databricks-bge-large-en'
VECTOR_SEARCH_ENDPOINT = 'vector_search'

SOURCE_NEWS_TABLE = f"{CATALOG}.{SCHEMA}.silver_news"
SOURCE_COMPANIES_TABLE = f"{CATALOG}.{SCHEMA}.silver_companies"
EMBEDDINGS_TABLE = f"{CATALOG}.{SCHEMA}.text_embeddings"
VECTOR_INDEX_NAME = f"{CATALOG}.{SCHEMA}.text_embeddings_index"

# Initialize Spark
try:
    spark = SparkSession.builder.getOrCreate()
except:
    print("Warning: SparkSession not available")

# Initialize MLflow client for embeddings
mlflow_client = mlflow.deployments.get_deploy_client("databricks")


def get_embeddings_batch(texts: list) -> list:
    """Call the embedding endpoint for a batch of texts."""
    response = mlflow_client.predict(
        endpoint=EMBEDDING_MODEL_ENDPOINT,
        inputs={"input": [t[:8000] for t in texts]}  # Truncate to model max
    )
    return [item["embedding"] for item in response["data"]]


# Define Pandas UDF for distributed embedding generation
@F.pandas_udf(ArrayType(FloatType()))
def generate_embeddings_udf(text_series: pd.Series) -> pd.Series:
    """Pandas UDF to generate embeddings in parallel across Spark workers.
    
    This runs on each Spark executor, processing batches of text in parallel.
    """
    # Convert series to list for batch processing
    texts = text_series.tolist()
    
    # Generate embeddings in batch (more efficient than one-by-one)
    try:
        embeddings = get_embeddings_batch(texts)
        return pd.Series(embeddings)
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        # Return empty embeddings on error
        return pd.Series([[0.0] * 1024] * len(texts))


def combine_text_corpus():
    """Combine news articles and company descriptions into a single corpus.
    
    Uses Spark DataFrame operations (no pandas collect on driver).
    """
    # News articles: title + summary + full_text
    df_news_text = (
        spark.table(SOURCE_NEWS_TABLE)
        .select(
            F.concat_ws(" | ", F.col("ticker"), F.col("title")).alias("doc_id"),
            F.lit("news").alias("doc_type"),
            F.col("ticker"),
            F.col("title"),
            F.coalesce(
                F.concat_ws("\\n\\n", F.col("title"), F.col("summary"), F.col("full_text")),
                F.concat_ws("\\n\\n", F.col("title"), F.col("summary")),
                F.col("title")
            ).alias("text"),
            F.col("published_at").cast("string").alias("metadata_date"),
            F.col("source").alias("metadata_source")
        )
        .filter(F.col("text").isNotNull())
    )
    
    # Company profiles: name + sector + description
    df_company_text = (
        spark.table(SOURCE_COMPANIES_TABLE)
        .select(
            F.col("ticker").alias("doc_id"),
            F.lit("company_profile").alias("doc_type"),
            F.col("ticker"),
            F.col("company_name").alias("title"),
            F.concat_ws("\\n\\n",
                F.col("company_name"),
                F.concat(F.lit("Sector: "), F.col("sector")),
                F.concat(F.lit("Industry: "), F.col("industry")),
                F.col("company_description")
            ).alias("text"),
            F.col("updated_at").cast("string").alias("metadata_date"),
            F.lit("Company Profile").alias("metadata_source")
        )
        .filter(F.col("text").isNotNull())
    )
    
    # Union both datasets
    df_corpus = df_news_text.union(df_company_text)
    
    print(f"Total documents to embed: {df_corpus.count()}")
    df_corpus.show(5, truncate=80)
    
    return df_corpus


def generate_embeddings_distributed(df_corpus):
    """Generate embeddings using Pandas UDF (distributed across Spark workers).
    
    This demonstrates true distributed Spark processing, not driver-side pandas.
    """
    print("\\nGenerating embeddings using distributed Pandas UDF...")
    
    # Apply Pandas UDF to generate embeddings in parallel
    df_embeddings = df_corpus.withColumn(
        "embedding", 
        generate_embeddings_udf(F.col("text"))
    )
    
    # Write to Delta table (idempotent overwrite)
    df_embeddings.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(EMBEDDINGS_TABLE)
    
    print(f"\\n✓ Embeddings table written: {EMBEDDINGS_TABLE}")
    print(f"  Total rows: {df_embeddings.count()}")
    print(f"  Embedding dimension: 1024 (BGE-large-en)")
    
    df_embeddings.select("doc_id", "doc_type", "ticker", "title").show(5, truncate=50)
    
    return df_embeddings


def create_vector_search_index():
    """Create Vector Search index with Delta Sync."""
    vsc = VectorSearchClient()
    
    # Enable Change Data Feed (required for Delta Sync)
    spark.sql(f"ALTER TABLE {EMBEDDINGS_TABLE} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")
    print(f"✓ Change Data Feed enabled on {EMBEDDINGS_TABLE}")
    
    # Create the index (Delta Sync mode - auto-syncs when table updates)
    try:
        index = vsc.create_delta_sync_index(
            endpoint_name=VECTOR_SEARCH_ENDPOINT,
            index_name=VECTOR_INDEX_NAME,
            source_table_name=EMBEDDINGS_TABLE,
            pipeline_type="TRIGGERED",
            primary_key="doc_id",
            embedding_dimension=1024,
            embedding_vector_column="embedding"
        )
        print(f"✓ Vector Search index created: {VECTOR_INDEX_NAME}")
    except Exception as e:
        if "already exists" in str(e):
            print(f"Index {VECTOR_INDEX_NAME} already exists. Syncing...")
        else:
            raise e


def test_semantic_search():
    """Test semantic search functionality."""
    vsc = VectorSearchClient()
    
    # Wait for index to be ready
    print("\\nWaiting for index to be ready...")
    max_wait = 300  # 5 minutes
    wait_interval = 15
    elapsed = 0
    
    while elapsed < max_wait:
        try:
            index = vsc.get_index(
                endpoint_name=VECTOR_SEARCH_ENDPOINT,
                index_name=VECTOR_INDEX_NAME
            )
            status = index.describe().get("status", {}).get("ready", False)
            if status:
                print(f"✓ Index is ready after {elapsed}s")
                break
        except Exception as e:
            if "is not ready" in str(e):
                print(f"  Index status: not ready yet... ({elapsed}s elapsed)")
            else:
                raise e
        time.sleep(wait_interval)
        elapsed += wait_interval
    
    # Test query
    test_query = "companies exposed to rising interest rates in the banking sector"
    print(f"\\nQuery: '{test_query}'\\n")
    
    # Generate query embedding
    query_emb = get_embeddings_batch([test_query])[0]
    
    # Search
    index = vsc.get_index(
        endpoint_name=VECTOR_SEARCH_ENDPOINT,
        index_name=VECTOR_INDEX_NAME
    )
    
    results = index.similarity_search(
        query_vector=query_emb,
        columns=["doc_id", "doc_type", "ticker", "title", "text"],
        num_results=5
    )
    
    print("Top results:")
    for i, row in enumerate(results.get("result", {}).get("data_array", []), 1):
        print(f"  {i}. [{row[2]}] {row[3]}")
        print(f"     Type: {row[1]} | Score: {row[-1]:.4f}\\n")
    
    return results


def run_pipeline():
    """Execute the complete embeddings pipeline."""
    print("=== Embeddings and Vector Search Pipeline ===")
    print(f"Model: {EMBEDDING_MODEL_ENDPOINT}")
    print(f"Vector Search Endpoint: {VECTOR_SEARCH_ENDPOINT}")
    print(f"Index: {VECTOR_INDEX_NAME}\\n")
    
    # Step 1: Combine text corpus
    print("[1/4] Combining text corpus from Silver tables...")
    df_corpus = combine_text_corpus()
    
    # Step 2: Generate embeddings (distributed)
    print("\\n[2/4] Generating embeddings with distributed Pandas UDF...")
    df_embeddings = generate_embeddings_distributed(df_corpus)
    
    # Step 3: Create Vector Search index
    print("\\n[3/4] Creating Vector Search index...")
    create_vector_search_index()
    
    # Step 4: Test semantic search
    print("\\n[4/4] Testing semantic search...")
    test_semantic_search()
    
    print("\\n✅ Pipeline complete!")


if __name__ == "__main__":
    run_pipeline()
