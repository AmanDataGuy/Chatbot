from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.store.base import BaseStore

from graph.state import ChatState
from prompts.system import build_system_prompt
from config import CHAT_MODEL


chat_llm = ChatGroq(model=CHAT_MODEL)


def make_chat_node(tools: list):
    """
    Returns a chat_node function with the given tools bound to the LLM.
    """
    llm_with_tools = chat_llm.bind_tools(tools) if tools else chat_llm

    def chat_node(state: ChatState, config: RunnableConfig, *, store: BaseStore):
        user_id = config["configurable"]["user_id"]
        ns = ("user", user_id, "details")

        # Load LTM
        items = store.search(ns)
        user_details = "\n".join(it.value.get("data", "") for it in items) if items else ""

        # Build system prompt (injects LTM + summary)
        system_msg = SystemMessage(
            content=build_system_prompt(user_details, state.get("summary", ""))
        )

        response = llm_with_tools.invoke([system_msg] + state["messages"])
        return {"messages": [response]}

    return chat_node