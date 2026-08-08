# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# MAGIC %md
# MAGIC # Generate Local Embeddings for Stock Research App
# MAGIC
# MAGIC This notebook generates embeddings using **sentence-transformers** (open-source) instead of Databricks Vector Search.
# MAGIC
# MAGIC ## Benefits:
# MAGIC - ✅ No OAuth permissions needed
# MAGIC - ✅ Faster (no API calls)
# MAGIC - ✅ Simpler deployment
# MAGIC - ✅ Works offline
# MAGIC
# MAGIC ## Steps:
# MAGIC 1. Install sentence-transformers
# MAGIC 2. Load company data from Unity Catalog (or create sample data)
# MAGIC 3. Generate embeddings using `all-MiniLM-L6-v2` model
# MAGIC 4. Save to `embeddings.pkl` in app directory
# MAGIC
# MAGIC ## Usage:
# MAGIC Run all cells to generate the embeddings file. Then deploy your app!
# MAGIC
# MAGIC ```bash
# MAGIC databricks apps deploy stock-market-assistant \
# MAGIC   --source-code-path /Workspace/Users/bchandra.ry@gmail.com/capstone-ai-stock-market \
# MAGIC   --mode SNAPSHOT
# MAGIC ```

# COMMAND ----------

# DBTITLE 1,Install dependencies
# MAGIC %pip install sentence-transformers numpy --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Import libraries and load model
from sentence_transformers import SentenceTransformer
import pandas as pd
import pickle
import numpy as np

# Load a small, fast model (80MB, good for demos)
print("Loading embedding model...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("✅ Model loaded!")

# COMMAND ----------

# DBTITLE 1,Load company data from Unity Catalog
# Using your existing catalog and schema from notebook 02
CATALOG = "stock_research_capstone"
SCHEMA = "main"

print(f"Loading data from {CATALOG}.{SCHEMA}...\n")

# Load company profiles from silver_companies
df_companies = spark.table(f"{CATALOG}.{SCHEMA}.silver_companies").toPandas()
print(f"✅ Loaded {len(df_companies)} companies")

# Prepare company data for embeddings
companies_data = []
for _, row in df_companies.iterrows():
    companies_data.append({
        'ticker': row['ticker'],
        'title': row['company_name'],
        'text': f"{row['company_name']}. Sector: {row['sector']}. Industry: {row['industry']}. {row.get('description', '')}",
        'date': '2024-01-01',
        'doc_type': 'company_profile'
    })

# Load news articles from silver_news
df_news = spark.table(f"{CATALOG}.{SCHEMA}.silver_news").toPandas()
print(f"✅ Loaded {len(df_news)} news articles")

# Prepare news data for embeddings
news_data = []
for _, row in df_news.iterrows():
    news_data.append({
        'ticker': row['ticker'],
        'title': row['title'],
        'text': f"{row['title']}. {row.get('summary', '')} {row.get('full_text', '')}",
        'date': str(row.get('published_at', '2024-01-01'))[:10],
        'doc_type': 'news_article'
    })

# Combine both datasets
all_data = companies_data + news_data
df = pd.DataFrame(all_data)

print(f"\n✅ Total documents for embedding: {len(df)}")
print(f"   - {len(companies_data)} company profiles")
print(f"   - {len(news_data)} news articles")
print(f"\nColumns: {df.columns.tolist()}")
df.head()

# COMMAND ----------

# DBTITLE 1,Generate embeddings for all documents
# Generate embeddings for each row
print("Generating embeddings...")

def generate_embedding(text):
    """Generate embedding for a single text."""
    if pd.isna(text) or text == '':
        return np.zeros(384)  # all-MiniLM-L6-v2 produces 384-dimensional embeddings
    return model.encode(str(text))

# Apply to text column
df['embedding'] = df['text'].apply(generate_embedding)

print(f"✅ Generated embeddings for {len(df)} documents")
print(f"Embedding dimension: {len(df['embedding'].iloc[0])}")

# Display sample
df[['ticker', 'title', 'text']].head()

# COMMAND ----------

# DBTITLE 1,Save embeddings to app directory
import os
from pyspark.sql.types import StringType, ArrayType, DoubleType
from pyspark.sql import functions as F

# Convert embeddings list to array for Spark
def embedding_to_list(emb):
    return emb.tolist() if hasattr(emb, 'tolist') else list(emb)

df['embedding_array'] = df['embedding'].apply(embedding_to_list)

# Save as Delta table
table_name = f"{CATALOG}.{SCHEMA}.document_embeddings"
df_spark = spark.createDataFrame(df[['ticker', 'title', 'text', 'date', 'doc_type', 'embedding_array']])
df_spark.write.format("delta").mode("overwrite").saveAsTable(table_name)

print(f"✅ Saved to Delta table: {table_name}")
print(f"   Rows: {len(df)}")

# Also save as pickle file for quick app loading
pickle_path = '/Workspace/Users/bchandra.ry@gmail.com/capstone-ai-stock-market/embeddings.pkl'
with open(pickle_path, 'wb') as f:
    pickle.dump(df[['ticker', 'title', 'text', 'date', 'doc_type', 'embedding']], f)

print(f"✅ Saved pickle file: {pickle_path}")
print(f"   File size: {os.path.getsize(pickle_path) / 1024:.2f} KB")

print("\n🎉 Done! Your app can now use local semantic search.")
print(f"\n💡 View the Delta table: spark.table('{table_name}').display()")

# COMMAND ----------

# DBTITLE 1,Verify Delta table
# Verify the Delta table was created successfully
table_name = f"{CATALOG}.{SCHEMA}.document_embeddings"

print(f"Table: {table_name}")
print(f"\nSchema:")
spark.table(table_name).printSchema()

print(f"\nSample data:")
display(spark.table(table_name).select('ticker', 'doc_type', 'title').limit(5))

print(f"\nRow count: {spark.table(table_name).count()}")