from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.store.base import BaseStore

from graph.state import ChatState
from tools.rag_tool import rag_tool
from prompts.system import build_system_prompt
from config import CHAT_MODEL


research_tools = [rag_tool]

research_llm = ChatGroq(model=CHAT_MODEL)
research_llm_with_tools = research_llm.bind_tools(research_tools)


def research_chat_node(state: ChatState, config: RunnableConfig, *, store: BaseStore):
    user_id = config["configurable"]["user_id"]
    ns = ("user", user_id, "details")

    items = store.search(ns)
    user_details = "\n".join(it.value.get("data", "") for it in items) if items else ""

    base_prompt = build_system_prompt(user_details, state.get("summary", ""))
    research_prompt = base_prompt + "\n\nYou have access to the user's uploaded documents. Use the rag_search tool to find relevant information from the documents and answer based on what you find. Only use the tool once per question."

    system_msg = SystemMessage(content=research_prompt)

    response = research_llm_with_tools.invoke([system_msg] + state["messages"])
    return {"messages": [response]}


research_tool_node = ToolNode(research_tools)

research_builder = StateGraph(ChatState)
research_builder.add_node("research_chat", research_chat_node)
research_builder.add_node("research_tools", research_tool_node)

research_builder.add_edge(START, "research_chat")
research_builder.add_conditional_edges("research_chat", tools_condition, {
    "tools": "research_tools",
    "__end__": END,
})
research_builder.add_edge("research_tools", "research_chat")

research_subgraph = research_builder.compile()