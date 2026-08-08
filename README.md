# AI Stock Market Research Assistant - Capstone Project

A complete AI-powered stock research platform combining Databricks Apps, Lakebase Postgres, Vector Search (RAG), and AI agents.

## 🎯 Features

* **Real-time Market Data**: Live stock prices, historical data, and fundamentals via Yahoo Finance (yfinance) API
* **Semantic Search**: Vector search over news articles and company profiles
* **Personal Watchlist**: Track your favorite stocks with OLTP storage in Lakebase
* **Research Notes**: Save and organize your analysis
* **AI Agent**: Intelligent assistant with 11 tools for market research
* **Interactive Dashboard**: Streamlit frontend with charts and analysis

## 📋 Architecture

```
┌─────────────────┐      ┌──────────────────┐      ┌─────────────────┐
│  Streamlit App  │─────▶│  AI Agent Tools  │─────▶│ Yahoo Finance   │
│   (Frontend)    │      │  (11 Functions)  │      │   (yfinance)    │
└─────────────────┘      └──────────────────┘      └─────────────────┘
         │                        │
         │                        │
         ▼                        ▼
┌─────────────────┐      ┌──────────────────┐
│   Lakebase      │      │  Vector Search   │
│   Postgres      │      │  (Embeddings)    │
│   (OLTP Data)   │      │  (Semantic RAG)  │
└─────────────────┘      └──────────────────┘
```

## 🚀 Setup Instructions

### 1. Prerequisites

* Databricks workspace (Free Edition supported)
* Unity Catalog enabled
* Yahoo Finance (yfinance) API key

### 2. Create Lakebase Project

```python
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.postgres import Project, ProjectSpec

w = WorkspaceClient()

op = w.postgres.create_project(
    project=Project(spec=ProjectSpec(
        display_name="Stock Research DB",
        pg_version=17
    )),
    project_id="stock-research-db"
)
project = op.wait()
print(f"✓ Project created: {project.name}")
```

### 3. Run the Notebooks in Order

#### Notebook 01: Create Postgres Tables
Creates the database schema in Lakebase (users, watchlists, research notes, etc.)

```bash
# Run cells 1-5 to set up the schema
```

#### Notebook 02: Data Ingestion Pipeline
Fetches data from Massive Stocks API and processes it into Delta tables.

```bash
# Configure your API key in cell 2
# Run all cells to ingest data
```

#### Notebook 03: Embeddings and Vector Index
Creates embeddings and sets up vector search for semantic retrieval.

```bash
# Update CATALOG, SCHEMA, and endpoints in cell 2
# Run all cells to create the index
```

#### Notebook 04: Agent
Defines the AI agent with all tools and capabilities.

```bash
# Update configuration in cell 2
# Run cell 17 to create database schema
# Run cell 18 to test agent capabilities
```

### 4. Configure the App

Update `app.yaml` with your values:

```yaml
env:
  - name: LAKEBASE_PROJECT_NAME
    value: "stock-research-db"  # Your project name
  
  - name: VECTOR_SEARCH_ENDPOINT
    value: "your-endpoint-name"
  
  - name: VECTOR_INDEX_NAME
    value: "catalog.schema.text_embeddings_index"
  
  - name: MASSIVE_STOCKS_API_KEY
    value: "your-api-key-here"
```

### 5. Deploy the App

```bash
# From the capstone-ai-stock-market directory
databricks apps create stock-research-assistant --source-code-path .

# Deploy
databricks apps deploy stock-research-assistant
```

### 6. Access the App

```bash
# Get the URL
databricks apps get stock-research-assistant
```

The app will be available at: `https://<workspace>.cloud.databricks.com/apps/stock-research-assistant`

## 📊 Using the App

### Chat Interface
Ask questions like:
* "Search for companies in the EV sector"
* "What's the current price of AAPL?"
* "Find news about interest rate exposure"

### Watchlist Management
* Add tickers using the sidebar form
* Click on a ticker to view detailed analysis
* Remove tickers with the ❌ button

### Stock Details View
* Real-time price metrics
* 90-day candlestick chart
* Relevant news and analysis (semantic search)
* Personal research notes

### Research Notes
* Save your analysis for each ticker
* Categorize notes (analysis, thesis, alert, other)
* Review past notes for any stock

## 🛠️ Development

### Local Testing

```bash
# Install dependencies
pip install streamlit psycopg2-binary pandas plotly requests mlflow databricks-sdk databricks-vectorsearch

# Run locally (requires Databricks auth)
streamlit run app.py
```

### Adding New Agent Tools

1. Define the tool in notebook `04_Agent`
2. Add it to the `tools` list in cell 13
3. Update the chat interface in `app.py` if needed

### Updating the Schema

Modify `setup_database_schema()` in notebook `04_Agent`, cell 17.

## 📁 Project Structure

```
capstone-ai-stock-market/
├── 01_postgres_tables.ipynb      # Database schema setup
├── 02_ingest_pipeline.ipynb      # Data ingestion (Spark)
├── 03_embed_and_index.ipynb      # Vector search setup
├── 04_Agent.ipynb                # AI agent definition
├── app.py                        # Streamlit frontend
├── app.yaml                      # Databricks App config
└── README.md                     # This file
```

## 🎓 Capstone Requirements Met

✅ **Data Pipeline in Spark**: Notebook 02 processes market data  
✅ **Third-party API**: Yahoo Finance (yfinance) API Stocks API integration  
✅ **Unstructured Data Processing**: Embeddings over news/company text  
✅ **Databricks App**: Streamlit frontend (app.py)  
✅ **AI Agent with Tools**: 11 tools for read/write operations  

## 🔧 Troubleshooting

### "Lakebase configuration missing"
* Check that `LAKEBASE_POSTGRES_HOST` is set in `app.yaml`
* Verify the project name matches your actual Lakebase project

### "Vector Search error"
* Ensure the index is created and synced (notebook 03)
* Verify endpoint and index names in `app.yaml`



### Connection timeouts
* Lakebase scales to zero after 5 minutes of inactivity
* First query after wake-up may take ~100ms

## 📚 Resources

* [Lakebase Documentation](https://docs.databricks.com/lakebase/)
* [Vector Search Guide](https://docs.databricks.com/vector-search/)
* [Databricks Apps](https://docs.databricks.com/apps/)

## 🤝 Contributing

This is a capstone project, but suggestions welcome:
1. Fork the project
2. Create your feature branch
3. Commit your changes
4. Open a pull request

## 📝 License

MIT License - See LICENSE file for details

## 👤 Author

Built as a Databricks capstone project demonstrating:
* Lakehouse architecture
* Real-time data integration
* Vector search / RAG
* AI agents with tool-calling
* Full-stack app development

---

**Need help?** Open an issue or check the Databricks Community forums.
