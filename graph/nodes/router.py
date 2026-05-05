from graph.state import ChatState

FINANCE_KEYWORDS = {
    "stock", "price", "buy", "purchase", "shares", "invest",
    "ticker", "market", "portfolio", "dividend", "aapl", "tsla",
}

RESEARCH_KEYWORDS = {
    "pdf", "document", "from the doc", "from the file", "in the pdf",
    "according to the", "explain from", "using the doc", "from my document",
    "what does the pdf", "summarize the pdf", "from the uploaded",
}


def router(state: ChatState) -> str:
    last_message = state["messages"][-1].content.lower()

    if any(kw in last_message for kw in FINANCE_KEYWORDS):
        return "finance"

    if any(kw in last_message for kw in RESEARCH_KEYWORDS):
        return "research"

    return "chat"