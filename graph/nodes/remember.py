import uuid
from langchain_core.runnables import RunnableConfig
from langgraph.store.base import BaseStore

from graph.state import ChatState
from memory.extractor import extract_memories


def remember_node(state: ChatState, config: RunnableConfig, *, store: BaseStore):
    """
    Reads existing LTM for the user, extracts new memories from the latest
    message, and writes only new atomic facts to PostgresStore.
    """
    user_id = config["configurable"]["user_id"]
    ns = ("user", user_id, "details")

    # Load existing memories
    items = store.search(ns)
    existing = "\n".join(it.value.get("data", "") for it in items) if items else "(empty)"

    # Latest user message
    last_text = state["messages"][-1].content

    # Extract new memories
    decision = extract_memories(last_text, existing)

    if decision.should_write:
        for mem in decision.memories:
            if mem.is_new and mem.text.strip():
                store.put(ns, str(uuid.uuid4()), {"data": mem.text.strip()})

    return {}