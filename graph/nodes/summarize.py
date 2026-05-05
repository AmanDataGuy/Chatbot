from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, RemoveMessage

from graph.state import ChatState
from config import CHAT_MODEL, SUMMARIZE_THRESHOLD


model = ChatGroq(model=CHAT_MODEL)


def should_summarize(state: ChatState) -> str:
    """
    Conditional edge: if messages exceed threshold, go to summarize.
    Otherwise go straight to router.
    """
    if len(state["messages"]) > SUMMARIZE_THRESHOLD:
        return "summarize"
    return "router"


def summarize_node(state: ChatState):
    """
    Summarizes the older messages and keeps only the last 2 verbatim.
    Extends existing summary if one already exists.
    """
    existing_summary = state.get("summary", "")

    if existing_summary:
        prompt = (
            f"Existing summary:\n{existing_summary}\n\n"
            "Extend the summary using the new conversation above."
        )
    else:
        prompt = "Summarize the conversation above."

    messages_for_summary = state["messages"] + [HumanMessage(content=prompt)]
    response = model.invoke(messages_for_summary)

    # Keep only last 2 messages verbatim, delete the rest
    messages_to_delete = state["messages"][:-2]
    delete_messages = [RemoveMessage(id=m.id) for m in messages_to_delete]

    return {"summary": response.content, "messages": delete_messages}