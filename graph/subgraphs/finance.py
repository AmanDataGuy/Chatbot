from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.store.base import BaseStore

from graph.state import ChatState
from tools.stock import stock_tools
from tools.web_search import web_search
from prompts.system import build_system_prompt
from config import CHAT_MODEL


finance_tools = stock_tools + [web_search]
finance_llm = ChatGroq(model=CHAT_MODEL)
finance_llm_with_tools = finance_llm.bind_tools(finance_tools)
finance_tool_node = ToolNode(finance_tools)


def finance_chat_node(state: ChatState, config: RunnableConfig, *, store: BaseStore):
    user_id = config["configurable"]["user_id"]
    ns = ("user", user_id, "details")

    items = store.search(ns)
    user_details = "\n".join(it.value.get("data", "") for it in items) if items else ""

    system_msg = SystemMessage(
        content=build_system_prompt(user_details, state.get("summary", ""))
    )

    response = finance_llm_with_tools.invoke([system_msg] + state["messages"])
    return {"messages": [response]}


finance_builder = StateGraph(ChatState)
finance_builder.add_node("finance_chat", finance_chat_node)
finance_builder.add_node("finance_tools", finance_tool_node)

finance_builder.add_edge(START, "finance_chat")
finance_builder.add_conditional_edges("finance_chat", tools_condition, {
    "tools": "finance_tools",
    "__end__": END,
})
finance_builder.add_edge("finance_tools", "finance_chat")

finance_subgraph = finance_builder.compile()