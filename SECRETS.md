# Secrets Management Guide

This project uses Databricks Secrets to securely store API keys and connection strings. **Never commit secrets to version control.**

## Quick Setup

### Step 1: Run the Setup Script

From your workspace, run:

```python
%run ./setup_secrets
```

Or from your local machine with Databricks CLI:

```bash
python setup_secrets.py
```

The script will prompt you for:
1. **Massive Stocks API key** - Get from [massiveapi.com](https://massiveapi.com)
2. **Lakebase project name** - Your Lakebase project (e.g., `new_database`)
3. **Lakebase branch** - Usually `production`
4. **Vector Search endpoint** (optional) - Can set later
5. **Vector Search index** (optional) - Can set later

### Step 2: Verify Secrets

List your secrets (values are redacted):

```bash
databricks secrets list-secrets stock-research
```

You should see:
```
massive-api-key
lakebase-project
lakebase-branch
lakebase-endpoint
vector-search-endpoint  # if you set it
vector-search-index     # if you set it
```

## Secrets Reference

| Secret Key | Description | Example Value |
|------------|-------------|---------------|
| `massive-api-key` | Massive Stocks API key | `k6918t...` |
| `lakebase-project` | Lakebase project name | `new_database` |
| `lakebase-branch` | Lakebase branch name | `production` |
| `lakebase-endpoint` | Full endpoint path | `projects/new_database/branches/production/endpoints/primary` |
| `vector-search-endpoint` | Vector Search endpoint | `vs-endpoint-stocks` |
| `vector-search-index` | Vector index full name | `stock_research_capstone.main.text_embeddings_index` |

## Using Secrets in Notebooks

```python
# Get API key
api_key = dbutils.secrets.get(scope="stock-research", key="massive-api-key")

# Get Lakebase config
project = dbutils.secrets.get(scope="stock-research", key="lakebase-project")
endpoint = dbutils.secrets.get(scope="stock-research", key="lakebase-endpoint")

# Get Vector Search config
vs_endpoint = dbutils.secrets.get(scope="stock-research", key="vector-search-endpoint")
vs_index = dbutils.secrets.get(scope="stock-research", key="vector-search-index")
```

## Using Secrets in Databricks Apps

The `app.yaml` file references secrets using `valueFrom.databricksSecret`:

```yaml
env:
  - name: MASSIVE_STOCKS_API_KEY
    valueFrom:
      databricksSecret:
        scope: stock-research
        key: massive-api-key
```

This automatically injects the secret value as an environment variable when the app starts.

## Security Best Practices

✅ **DO:**
* Use `dbutils.secrets.get()` to read secrets
* Store all sensitive data in Databricks Secrets
* Use the `setup_secrets.py` script for consistent setup
* Keep secrets in a separate scope (`stock-research`)

❌ **DON'T:**
* Print secret values (they'll be redacted, but avoid the habit)
* Hardcode API keys in notebooks or files
* Commit secrets to Git
* Share secret values in chat or documentation
* Use `getpass()` in production notebooks (use secrets instead)

## Updating Secrets

To update a secret:

```python
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()
w.secrets.put_secret(
    scope="stock-research",
    key="massive-api-key",
    string_value="new-api-key-value"
)
```

Or re-run `setup_secrets.py` to update all secrets.

## Deleting Secrets

To delete a secret:

```python
w.secrets.delete_secret(scope="stock-research", key="massive-api-key")
```

To delete the entire scope:

```python
w.secrets.delete_scope(scope="stock-research")
```

## Troubleshooting

### "Secret does not exist with scope: stock-research"

**Solution**: Run `setup_secrets.py` to create the secrets.

### "PERMISSION_DENIED: User does not have MANAGE permission"

**Solution**: You need admin permissions to create secret scopes. Ask your workspace admin, or use a personal access token scope (if enabled).

### "Secret value is empty"

**Solution**: The secret exists but has no value. Re-run `setup_secrets.py` to set it.

### App can't read secrets

**Solution**: Verify the app.yaml uses the correct scope name (`stock-research`) and key names.

## Migration from Hardcoded Values

If you previously hardcoded values:

1. Run `setup_secrets.py` to store them in secrets
2. Update your notebooks to use `dbutils.secrets.get()`
3. Update `app.yaml` to use `valueFrom.databricksSecret`
4. Delete hardcoded values from your code
5. **Important**: If you committed hardcoded secrets, rotate them immediately

## Where Secrets Are Used

| File | Usage |
|------|-------|
| `02_ingest_pipeline.ipynb` (Cell 1) | Massive API key |
| `02_ingest_pipeline.ipynb` (Cell 9) | Lakebase connection |
| `03_embed_and_index.ipynb` | Vector Search config |
| `04_Agent.ipynb` (Cell 3) | All secrets |
| `app.yaml` | All secrets (injected as env vars) |
| `app.py` | Reads from `os.getenv()` (populated by app.yaml) |

## Next Steps

After setting up secrets:

1. ✅ Run [02_ingest_pipeline](#notebook-2626010614797196) to fetch market data
2. ✅ Run [03_embed_and_index](#notebook-2880997541649395) to create vector index
3. ✅ Run [04_Agent](#notebook-3917973374213396) to test agent tools
4. ✅ Deploy your app: `databricks apps deploy stock-research-assistant`

---

**Remember**: Secrets are workspace-scoped. If you move to a different workspace, re-run `setup_secrets.py`.
