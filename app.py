import os
import streamlit as st
from langchain_core.messages import HumanMessage
from langgraph.types import Command
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.postgres import PostgresStore

from config import DB_URI, DOCS_DIR
from mcpclient.client import get_mcp_tools
from graph.builder import build_graph
from memory.checkpointer import get_checkpointer
from rag.vectorstore import rebuild_vectorstore
from memory.store import get_store

# ----------------------------------------------------------------
# Page config
# ----------------------------------------------------------------
st.set_page_config(
    page_title="Chatbot",
    page_icon="🤖",
    layout="wide",
)

# ----------------------------------------------------------------
# Init Postgres + graph once per app session
# ----------------------------------------------------------------
@st.cache_resource
def init_chatbot():
    mcp_tools = get_mcp_tools()
    checkpointer = get_checkpointer()
    store = get_store()
    chatbot = build_graph(mcp_tools, checkpointer, store)
    return chatbot, store

chatbot, store = init_chatbot()


# ----------------------------------------------------------------
# Session state defaults
# ----------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []          # display history

if "pending_interrupt" not in st.session_state:
    st.session_state.pending_interrupt = None   # HITL prompt string

if "user_id" not in st.session_state:
    st.session_state.user_id = "u1"

if "thread_id" not in st.session_state:
    st.session_state.thread_id = f"thread-{st.session_state.user_id}"

with st.sidebar:

# ---- Stack info ----
    st.markdown("⚡ **Stack**")
    st.caption("LangGraph · Groq LLaMA 3.3 · PostgreSQL · FAISS")

    st.markdown("🔌 **Tools**")
    st.caption("• DuckDuckGo Search\n• Calculator\n• Stock Price\n• RAG")

    st.markdown("🧠 **Long-term Memory**")
    st.caption("Active via PostgreSQL")

    st.divider()

    # ---- Session ----
    st.markdown("👤 **Session**")
    user_id = st.text_input("User ID", value=st.session_state.user_id)
    if user_id != st.session_state.user_id:
        import time
        st.session_state.user_id = user_id
        st.session_state.thread_id = f"thread-{user_id}-{int(time.time())}"  # add this line
        st.session_state.messages = []
        st.session_state.pending_interrupt = None
        st.rerun()

    if st.button("🆕 New Chat"):
        import time
        st.session_state.thread_id = f"thread-{user_id}-{int(time.time())}"
        st.session_state.messages = []
        st.session_state.pending_interrupt = None
        st.rerun()

    st.divider()

    # ---- PDF uploader ----
    st.markdown("📄 **Knowledge Base**")
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
    if uploaded_file is not None:
        os.makedirs(DOCS_DIR, exist_ok=True)
        save_path = os.path.join(DOCS_DIR, uploaded_file.name)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.read())
        with st.spinner("Indexing PDF..."):
            rebuild_vectorstore()
        st.success(f"'{uploaded_file.name}' indexed!")

    st.divider()

    # ---- LTM memory viewer ----
    st.markdown("💾 **Stored Memories**")
    ns = ("user", st.session_state.user_id, "details")
    items = store.search(ns)
    if items:
        for item in items:
            st.markdown(f"- {item.value.get('data', '')}")
    else:
        st.caption("No memories yet.")
# ----------------------------------------------------------------
# Main chat area
# ----------------------------------------------------------------
st.title("🤖 Chatbot")
st.caption("Powered by LangGraph · Groq · Postgres · FAISS · MCP")

# Build config from current user
config = {
    "configurable": {
        "thread_id": st.session_state.thread_id,
        "user_id": st.session_state.user_id,
    }
}

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# ----------------------------------------------------------------
# HITL approval block — shown when graph is paused
# ----------------------------------------------------------------
if st.session_state.pending_interrupt:
    st.warning(f"⚠️ **Approval Required:** {st.session_state.pending_interrupt}")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("✅ Approve", use_container_width=True):
            with st.spinner("Resuming..."):
                result = chatbot.invoke(Command(resume="yes"), config=config)

            st.session_state.pending_interrupt = None
            response = result["messages"][-1].content
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()

    with col2:
        if st.button("❌ Deny", use_container_width=True):
            with st.spinner("Resuming..."):
                result = chatbot.invoke(Command(resume="no"), config=config)

            st.session_state.pending_interrupt = None
            response = result["messages"][-1].content
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()


# ----------------------------------------------------------------
# Chat input
# ----------------------------------------------------------------
if prompt := st.chat_input("Ask me anything...", disabled=st.session_state.pending_interrupt is not None):

    # Show user message immediately
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Invoke graph
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = chatbot.invoke(
                {"messages": [HumanMessage(content=prompt)]},
                config=config,
            )

        # Check for HITL interrupt
        interrupts = result.get("__interrupt__", [])

        if interrupts:
            st.session_state.pending_interrupt = interrupts[0].value
            st.rerun()
        else:
            response = result["messages"][-1].content
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.markdown(response)