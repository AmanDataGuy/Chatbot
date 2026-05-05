import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from config import GITHUB_TOKEN


async def _fetch_tools():
    servers = {
        "duckduckgo": {
            "transport": "stdio",
            "command": "cmd",
            "args": ["/c", "uvx", "duckduckgo-mcp"],
        }
    }

    if GITHUB_TOKEN:
        servers["github"] = {
            "transport": "stdio",
            "command": "cmd",
            "args": ["/c", "npx", "-y", "@modelcontextprotocol/server-github"],
            "env": {"GITHUB_PERSONAL_ACCESS_TOKEN": GITHUB_TOKEN},
        }

    client = MultiServerMCPClient(servers)
    tools = await client.get_tools()
    print(f"Loaded {len(tools)} MCP tools from: {list(servers.keys())}")
    return tools


def get_mcp_tools() -> list:
    return []