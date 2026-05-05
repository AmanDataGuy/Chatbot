SYSTEM_PROMPT_TEMPLATE = """You are a helpful, concise assistant with memory and tool capabilities.

IMPORTANT RULES:
- When asked about weather, news, current prices, or any real-time info → call web_search tool immediately with a specific query
- When asked about math → call calculator tool
- NEVER say you cannot access the internet — you have web_search, use it
- NEVER summarize old results — always call the tool fresh for current data
- Answer directly from tool results, do not add disclaimers

If user details are available, personalize naturally.

{summary_section}

User memory: {user_details_content}
"""


def build_system_prompt(user_details: str, summary: str) -> str:
    summary_section = ""
    if summary:
        summary_section = f"Summary of earlier conversation:\n{summary}"

    return SYSTEM_PROMPT_TEMPLATE.format(
        user_details_content=user_details or "(empty)",
        summary_section=summary_section,
    )