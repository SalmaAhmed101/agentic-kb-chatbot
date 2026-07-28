"""
app.py
Streamlit chat interface for the Etisalat Egypt Agentic Knowledge-Base Chatbot.

Run with:
    streamlit run src/app.py
"""

import os
import sys

# allow running as `streamlit run src/app.py` from repo root
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
from graph import ask

st.set_page_config(page_title="Etisalat Egypt KB Chatbot", page_icon="📶")
st.title("📶 Etisalat Egypt Knowledge-Base Chatbot")
st.caption("RAG chatbot over Etisalat Egypt mobile plans — built with LangGraph + ChromaDB")

if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {"role", "content"}

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask about Etisalat Egypt plans (Arabic or English)..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching the knowledge base..."):
            # last 6 turns as history, excluding the message we just added
            history = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages[-7:-1]
            ]
            try:
                result = ask(prompt, chat_history=history)
                answer = result["answer"]
            except Exception as e:
                answer = (
                    f"⚠️ Something went wrong calling the model/vector store: `{e}`\n\n"
                    "Check that `OPENAI_API_KEY` is set and that you've run "
                    "`python src/ingest.py` to build the ChromaDB index."
                )
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})

with st.sidebar:
    st.header("About")
    st.markdown(
        "This chatbot answers questions about Etisalat Egypt's:\n"
        "- Data Line packages\n"
        "- Prepaid systems (Aqwa Kart, Ahlan, etc.)\n"
        "- Emerald postpaid family plans\n"
        "- Hekaya Mix & Hekaya Internet packages\n"
    )
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()
