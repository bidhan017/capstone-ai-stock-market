"""
One-time setup script: creates the Databricks secret scope and stores all
secrets needed for the stock research capstone project.

Run this locally (with the Databricks CLI configured) or from a notebook.
NEVER commit the resulting secret values anywhere.

Secrets stored:
- massive-api-key: Your Massive Stocks API key
- lakebase-project: Lakebase project name
- lakebase-host: Lakebase Postgres host
- lakebase-endpoint: Lakebase endpoint name for OAuth token generation

Usage:
    python setup_secrets.py
"""
from databricks.sdk import WorkspaceClient
import getpass

def setup_secrets():
    w = WorkspaceClient()
    
    scope_name = "stock-research"
    
    # Create scope (if it doesn't exist)
    try:
        w.secrets.create_scope(scope=scope_name)
        print(f"✓ Created secret scope: {scope_name}")
    except Exception as e:
        if "already exists" in str(e).lower():
            print(f"✓ Secret scope '{scope_name}' already exists")
        else:
            raise
    
    print("\n" + "="*60)
    print("STOCK RESEARCH CAPSTONE - SECRET SETUP")
    print("="*60)
    
    # Massive Stocks API Key
    print("\n1. Massive Stocks API")
    print("   Get your free API key at: https://massiveapi.com")
    api_key = getpass.getpass("   Paste your Massive API key: ")
    
    if api_key:
        w.secrets.put_secret(
            scope=scope_name,
            key="massive-api-key",
            string_value=api_key
        )
        print("   ✓ Stored: massive-api-key")
    
    # Lakebase Configuration
    print("\n2. Lakebase Configuration")
    print("   Your Lakebase project details:")
    
    lakebase_project = input("   Lakebase project name (e.g., new_database): ").strip() or "new_database"
    w.secrets.put_secret(
        scope=scope_name,
        key="lakebase-project",
        string_value=lakebase_project
    )
    print(f"   ✓ Stored: lakebase-project = {lakebase_project}")
    
    lakebase_branch = input("   Lakebase branch name (default: production): ").strip() or "production"
    w.secrets.put_secret(
        scope=scope_name,
        key="lakebase-branch",
        string_value=lakebase_branch
    )
    print(f"   ✓ Stored: lakebase-branch = {lakebase_branch}")
    
    # Construct endpoint name
    endpoint_name = f"projects/{lakebase_project}/branches/{lakebase_branch}/endpoints/primary"
    w.secrets.put_secret(
        scope=scope_name,
        key="lakebase-endpoint",
        string_value=endpoint_name
    )
    print(f"   ✓ Stored: lakebase-endpoint = {endpoint_name}")
    
    # Vector Search (optional - can be set later)
    print("\n3. Vector Search Configuration (optional - press Enter to skip)")
    vector_endpoint = input("   Vector Search endpoint name: ").strip()
    if vector_endpoint:
        w.secrets.put_secret(
            scope=scope_name,
            key="vector-search-endpoint",
            string_value=vector_endpoint
        )
        print(f"   ✓ Stored: vector-search-endpoint")
    
    vector_index = input("   Vector Search index name (e.g., catalog.schema.index): ").strip()
    if vector_index:
        w.secrets.put_secret(
            scope=scope_name,
            key="vector-search-index",
            string_value=vector_index
        )
        print(f"   ✓ Stored: vector-search-index")
    
    print("\n" + "="*60)
    print("✓ SECRET SETUP COMPLETE!")
    print("="*60)
    print(f"\nSecret scope: {scope_name}")
    print("\nTo view your secrets (values are redacted):")
    print(f"    databricks secrets list-secrets {scope_name}")
    print("\nTo use in notebooks:")
    print(f'    dbutils.secrets.get(scope="{scope_name}", key="massive-api-key")')
    print("\n⚠️  Security reminder:")
    print("    - Never print secret values")
    print("    - Never commit secrets to version control")
    print("    - Use dbutils.secrets.get() in all notebooks")

if __name__ == "__main__":
    setup_secrets()
