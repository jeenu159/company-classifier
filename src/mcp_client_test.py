import asyncio
import sys
from langchain_mcp_adapters.client import MultiServerMCPClient


async def main():
    client = MultiServerMCPClient({
        "company-intel": {
            "command": sys.executable,          # the venv's python, so the server gets the same packages
            "args": ["src/mcp_server.py"],
            "transport": "stdio",
        }
    })
    tools = await client.get_tools()
    for t in tools:
        print(f"TOOL: {t.name} - {t.description[:90]}")

    fda = next(t for t in tools if t.name == "fda_profile")
    result = await fda.ainvoke({"company": "Boiron"})
    print("\nRESULT (first 500 chars):")
    print(str(result)[:500])


asyncio.run(main())