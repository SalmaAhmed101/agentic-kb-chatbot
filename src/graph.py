"""
graph.py
The agentic core of the chatbot: a LangGraph state machine with four nodes
matching the architecture in the project brief (section 5):

    understand -> retrieve -> generate -> respond

Each node reads/writes a shared AgentState. Conditional routing sends the
flow to a "no_results" fallback if retrieval comes back empty, instead of
letting the LLM hallucinate an answer.
"""

import os
from typing import TypedDict, List, Dict, Any, Optional

from openai import OpenAI
import chromadb
from chromadb.utils import embedding_functions
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv

load_dotenv()

CHROMA_DIR = os.path.join(os.path.dirname(__file__), "..", "chroma_db")
COLLECTION_NAME = "etisalat_kb"
TOP_K = 5

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL", "https://api.groq.com/openai/v1"),
)
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")


class AgentState(TypedDict):
    question: str
    chat_history: List[Dict[str, str]]
    normalized_query: str
    retrieved: List[Dict[str, Any]]
    answer: str
    sources: List[str]


_embed_fn = None

def get_embed_fn():
    global _embed_fn
    if _embed_fn is None:
        _embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="paraphrase-multilingual-MiniLM-L12-v2"
        )
    return _embed_fn

def get_collection():
    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
    return chroma_client.get_collection(name=COLLECTION_NAME, embedding_function=get_embed_fn())


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def understand_node(state: AgentState) -> AgentState:
    """
    Lightweight query-understanding step: expands abbreviations / normalizes
    the query so retrieval works even for short or vague questions.
    Uses the LLM only for this cheap rewriting task, not the final answer.
    """
    question = state["question"]
    try:
        resp = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You rewrite user questions about Etisalat Egypt telecom plans "
                        "into a clear, standalone search query. Keep the original language "
                        "(Arabic or English). Do not answer the question, only rewrite it. "
                        "Return only the rewritten query, nothing else."
                    ),
                },
                {"role": "user", "content": question},
            ],
            temperature=0,
            max_tokens=100,
        )
        normalized = resp.choices[0].message.content.strip()
    except Exception:
        normalized = question  # fail open: just use the raw question

    state["normalized_query"] = normalized or question
    return state


def retrieve_node(state: AgentState) -> AgentState:
    collection = get_collection()
    results = collection.query(
        query_texts=[state["normalized_query"]],
        n_results=TOP_K,
    )

    retrieved = []
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0]

    for doc, meta, dist in zip(docs, metas, dists):
        retrieved.append({"text": doc, "metadata": meta, "distance": dist})

    state["retrieved"] = retrieved
    return state


def route_after_retrieve(state: AgentState) -> str:
    """Conditional edge: no chunks (or all low-relevance) -> fallback."""
    if not state["retrieved"]:
        return "no_results"
    # Chroma distances are lower = more similar (cosine distance here).
    best_distance = min(r["distance"] for r in state["retrieved"])
    if best_distance > 0.8:
        return "no_results"
    return "generate"


def generate_node(state: AgentState) -> AgentState:
    context_blocks = []
    for i, r in enumerate(state["retrieved"], start=1):
        meta = r["metadata"]
        context_blocks.append(
            f"[Source {i}: {meta.get('title')} — {meta.get('heading_path')}]\n{r['text']}"
        )
    context = "\n\n".join(context_blocks)

    system_prompt = (
        "You are the Etisalat Egypt customer-support assistant. "
        "Answer ONLY using the provided context below. "
        "If the context does not fully answer the question, say what you do "
        "know and note what's missing — never invent prices, package names, or numbers. "
        "Answer in the same language the user asked in (Arabic or English). "
        "Keep answers concise and cite which source(s) you used by number, e.g. (Source 2)."
    )
    user_prompt = f"Context:\n{context}\n\nQuestion: {state['question']}"

    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            *state.get("chat_history", []),
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=600,
    )
    state["answer"] = resp.choices[0].message.content.strip()
    state["sources"] = sorted({r["metadata"]["source"] for r in state["retrieved"]})
    return state


def no_results_node(state: AgentState) -> AgentState:
    state["answer"] = (
        "I couldn't find this in the Etisalat Egypt knowledge base I have access to. "
        "Could you rephrase, or ask about mobile plans, data lines, prepaid systems, "
        "or Emerald postpaid packages?"
    )
    state["sources"] = []
    return state


def respond_node(state: AgentState) -> AgentState:
    """Final formatting step: appends source citations to the answer."""
    if state["sources"]:
        cited = ", ".join(state["sources"])
        state["answer"] = f"{state['answer']}\n\n_Sources: {cited}_"
    return state


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("understand", understand_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    graph.add_node("no_results", no_results_node)
    graph.add_node("respond", respond_node)

    graph.set_entry_point("understand")
    graph.add_edge("understand", "retrieve")
    graph.add_conditional_edges(
        "retrieve",
        route_after_retrieve,
        {"generate": "generate", "no_results": "no_results"},
    )
    graph.add_edge("generate", "respond")
    graph.add_edge("no_results", "respond")
    graph.add_edge("respond", END)

    return graph.compile()


_compiled_graph = None


def get_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


def ask(question: str, chat_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
    app = get_graph()
    initial_state: AgentState = {
        "question": question,
        "chat_history": chat_history or [],
        "normalized_query": "",
        "retrieved": [],
        "answer": "",
        "sources": [],
    }
    final_state = app.invoke(initial_state)
    return {"answer": final_state["answer"], "sources": final_state["sources"]}


if __name__ == "__main__":
    # Quick manual test from the command line
    q = input("Ask a question about Etisalat Egypt plans: ")
    result = ask(q)
    print("\n" + result["answer"])
