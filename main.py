from langchain_core.messages import HumanMessage
from langgraph.types import Command

from mcpclient.client import get_mcp_tools
from graph.builder import build_graph
from memory.checkpointer import get_checkpointer
from memory.store import get_store


def main():
    print("Starting chatbot...")

    # 1. Fetch remote MCP tools at startup
    mcp_tools = get_mcp_tools()

    # 2. Open Postgres connections for STM (checkpointer) and LTM (store)
    checkpointer = get_checkpointer()
    store = get_store()

    # 3. Build and compile the graph
    chatbot = build_graph(mcp_tools, checkpointer, store)

    print("Chatbot ready! Type 'exit' to quit.\n")

    # 4. Ask for user_id once per session
    user_id = input("Enter your user ID (e.g. u1): ").strip() or "u1"
    thread_id = f"thread-{user_id}"

    config = {
        "configurable": {
            "thread_id": thread_id,
            "user_id": user_id,
        }
    }

    # 5. Conversation loop
    while True:
        user_input = input("You: ").strip()

        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        if not user_input:
            continue

        state = {"messages": [HumanMessage(content=user_input)]}

        # Run the graph — may pause on HITL interrupt
        result = chatbot.invoke(state, config=config)

        # Check for HITL interrupt (purchase_stock approval)
        interrupts = result.get("__interrupt__", [])

        if interrupts:
            prompt_to_human = interrupts[0].value
            print(f"\nApproval needed: {prompt_to_human}")
            decision = input("Your decision (yes/no): ").strip().lower()

            # Resume graph with human decision
            result = chatbot.invoke(
                Command(resume=decision),
                config=config,
            )

        last_msg = result["messages"][-1]
        print(f"\nBot: {last_msg.content}\n")


if __name__ == "__main__":
    main()