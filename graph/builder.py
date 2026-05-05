from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition

from graph.state import ChatState
from graph.nodes.remember import remember_node
from graph.nodes.chat import make_chat_node
from graph.nodes.summarize import summarize_node, should_summarize
from graph.nodes.router import router
from graph.subgraphs.finance import finance_subgraph
from graph.subgraphs.research import research_subgraph
from tools.calculator import calculator
from tools.web_search import web_search


def build_graph(mcp_tools: list, checkpointer, store):
    """
    Assembles and compiles the main chatbot graph.

    Parameters
    ----------
    mcp_tools   : list of tools fetched from remote MCP servers
    checkpointer: PostgresSaver instance (STM - per thread persistence)
    store       : PostgresStore instance (LTM - per user persistence)
    """

    # General chat gets: calculator + all MCP tools
    general_tools = [calculator, web_search] + mcp_tools
    general_chat_node = make_chat_node(general_tools)
    general_tool_node = ToolNode(general_tools)


    # ----------------------------------------------------------------
    # Build main graph
    # ----------------------------------------------------------------
    graph = StateGraph(ChatState)

    # Nodes
    graph.add_node("remember", remember_node)
    graph.add_node("summarize", summarize_node)
    graph.add_node("finance", finance_subgraph)
    graph.add_node("research", research_subgraph)
    graph.add_node("chat", general_chat_node)
    graph.add_node("tools", general_tool_node)

    # Edges
    graph.add_edge(START, "remember")

    # After remember: check if we need to summarize
    graph.add_conditional_edges("remember", should_summarize, {
        "summarize": "summarize",
        "router": "router",
    })

    graph.add_node("router", lambda state: {})   # pass-through node
    graph.add_conditional_edges("router", router, {
        "finance": "finance",
        "research": "research",
        "chat": "chat",
    })

    graph.add_edge("summarize", "router")

    graph.add_edge("finance", END)
    graph.add_edge("research", END)

    # General chat tool loop
    graph.add_conditional_edges("chat", tools_condition, {
        "tools": "tools",
        "__end__": END,
    })
    graph.add_edge("tools", "chat")

    chatbot = graph.compile(checkpointer=checkpointer, store=store)
    return chatbot