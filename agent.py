from __future__ import annotations

import ast
import math
import os
import re
from datetime import datetime
from typing import Any, Dict, List, TypedDict

import chromadb
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from sentence_transformers import SentenceTransformer

try:
    from langchain_groq import ChatGroq
except Exception:
    ChatGroq = None  # type: ignore


class CapstoneState(TypedDict, total=False):
    question: str
    messages: List[Dict[str, str]]
    route: str
    retrieved: str
    sources: List[str]
    tool_result: str
    answer: str
    faithfulness: float
    eval_retries: int
    employee_name: str

from knowledge_base.hr_docs import DOCUMENTS

embedder = SentenceTransformer("all-MiniLM-L6-v2")
_texts = [d["text"] for d in DOCUMENTS]
_ids = [d["id"] for d in DOCUMENTS]
_metas = [{"topic": d["topic"]} for d in DOCUMENTS]
_embeddings = embedder.encode(_texts, normalize_embeddings=True)

_client = chromadb.Client()
collection = _client.get_or_create_collection(name="hr_policy_assistant")

try:
    peek = collection.get(include=[])
    existing_ids = set(peek.get("ids", []))
except Exception:
    existing_ids = set()

if not existing_ids:
    collection.add(
        ids=_ids,
        documents=_texts,
        embeddings=_embeddings.tolist(),
        metadatas=_metas,
    )

_llm = None
if ChatGroq is not None and os.getenv("GROQ_API_KEY"):
    _llm = ChatGroq(
        model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
        temperature=0,
    )


def _llm_text(prompt: str) -> str:
    if _llm is None:
        return ""
    try:
        result = _llm.invoke(prompt)
        if hasattr(result, "content"):
            return str(result.content).strip()
        return str(result).strip()
    except Exception:
        return ""


def _fallback_route(question: str) -> str:
    q = question.lower().strip()
    if any(k in q for k in ["date", "time", "today", "tomorrow", "day of week", "what day"]):
        return "tool"
    if re.search(r"\d+\s*[\+\-\*/]\s*\d+", q):
        return "tool"
    if any(k in q for k in ["my name is", "call me", "i am "]):
        return "skip"
    return "retrieve"


def _fallback_faithfulness(answer: str, retrieved: str, tool_result: str) -> float:
    if not retrieved:
        return 1.0
    answer_words = set(re.findall(r"[a-zA-Z]+", answer.lower()))
    context_words = set(re.findall(r"[a-zA-Z]+", (retrieved + " " + tool_result).lower()))
    if not answer_words:
        return 0.0
    overlap = len(answer_words & context_words) / max(len(answer_words), 1)
    return round(min(1.0, max(0.0, 0.4 + overlap)), 2)


def _safe_math_eval(expr: str) -> float:
    allowed_nodes = (
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.Constant,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.FloorDiv,
        ast.Pow,
        ast.Mod,
        ast.USub,
        ast.UAdd,
        ast.Load,
    )

    def _check(node: ast.AST) -> None:
        if not isinstance(node, allowed_nodes):
            raise ValueError(f"Unsupported expression element: {type(node).__name__}")
        for child in ast.iter_child_nodes(node):
            _check(child)

    tree = ast.parse(expr, mode="eval")
    _check(tree)
    return eval(compile(tree, "<expr>", "eval"), {"__builtins__": {}}, {"math": math})

import re

def memory_node(state):
    question = state.get("question", "")
    messages = state.get("messages", [])

    state["tool_result"] = ""

    messages.append({"role": "user", "content": question})
    messages = messages[-6:]

    match = re.search(r"my name is (\w+)", question.lower())
    if match:
        state["employee_name"] = match.group(1).capitalize()

    state["messages"] = messages
    return state
    
def router_node(state):
    q = state.get("question", "").lower()

    if "what did i tell you" in q:
        state["route"] = "skip"
        return state

    if any(word in q for word in ["calculate", "date", "time"]):
        state["route"] = "tool"
        return state

    if any(word in q for word in ["recipe", "cook", "food", "chicken"]):
        state["route"] = "skip"
        return state

    if any(word in q for word in ["ignore", "system prompt", "instructions"]):
        state["route"] = "skip"
        return state

    if "my name is" in q and len(q.split()) > 5:
        state["route"] = "retrieve"
        return state

    state["route"] = "retrieve"
    return state

def retrieval_node(state):
    question = state["question"]

    emb = embedder.encode([question]).tolist()

    res = collection.query(query_embeddings=emb, n_results=3)

    docs = res["documents"][0]
    metas = res["metadatas"][0]

    context = ""
    sources = []

    for doc, meta in zip(docs, metas):
        context += f"[{meta['topic']}] {doc}\n"
        sources.append(meta["topic"])

    state["retrieved"] = context.strip()
    state["sources"] = sources

    return state


def skip_retrieval_node(state):
    state["retrieved"] = ""
    state["sources"] = []
    return state


def tool_node(state):
    question = state["question"].lower()

    import datetime
    import re

    if "date" in question or "time" in question:
        now = datetime.datetime.now()
        state["tool_result"] = f"Today is {now.strftime('%A, %d %B %Y')}. Current time is {now.strftime('%I:%M %p')}."
        return state


    match = re.search(r"(\d+)\s*([\+\-\*/])\s*(\d+)", question)

    if match:
        a, op, b = match.groups()
        a, b = int(a), int(b)

        if op == "+": result = a + b
        elif op == "-": result = a - b
        elif op == "*": result = a * b
        elif op == "/": result = a / b

        state["tool_result"] = f"The result of {a} {op} {b} is {result}"
        return state

    state["tool_result"] = "Tool error"
    return state


def answer_node(state: CapstoneState) -> CapstoneState:
    retrieved = state.get("retrieved", "")
    tool_result = state.get("tool_result", "")
    history = state.get("messages", [])
    question = state.get("question", "")
    employee_name = state.get("employee_name", "")

    if tool_result:
        state["answer"] = tool_result
        return state

    if "what did i tell you" in question.lower():
        if employee_name:
            state["answer"] = f"You told me your name is {employee_name}."
        else:
            state["answer"] = "I do not have your name yet."
        return state

    if not retrieved:
        state["answer"] = "I do not know. This is outside HR policy scope."
        return state

    system_prompt = """
You are an HR Policy Assistant.
Use ONLY the provided policy context.
Never invent policy details.
If unsure, say you do not know.
Keep answer short and clear.
""".strip()

    history_text = "\n".join(f"{m['role']}: {m['content']}" for m in history[-6:])

    if _llm:
        prompt = f"""
SYSTEM:
{system_prompt}

EMPLOYEE NAME:
{employee_name or "Not provided"}

POLICY CONTEXT:
{retrieved}

CONVERSATION HISTORY:
{history_text or "No prior history."}

QUESTION:
{question}
""".strip()

        answer = _llm_text(prompt)

        if not answer:
            answer = retrieved[:300]
    else:
        answer = retrieved[:300]

    if employee_name:
        answer = f"{employee_name}, {answer}"

    state["answer"] = answer
    return state


def eval_node(state: CapstoneState) -> CapstoneState:
    retrieved = state.get("retrieved", "")
    tool_result = state.get("tool_result", "")
    answer = state.get("answer", "")

    if tool_result:
        state["faithfulness"] = 1.0
        return state

    if not retrieved:
        state["faithfulness"] = 1.0
        return state

    prompt = f"""
Rate the faithfulness of this answer to the provided context from 0.0 to 1.0.
Return only a number.

CONTEXT:
{retrieved}

ANSWER:
{answer}
""".strip()

    raw = _llm_text(prompt)

    score = None
    try:
        score = float(re.findall(r"\d+(?:\.\d+)?", raw)[0])
    except Exception:
        score = None

    if score is None:
        score = _fallback_faithfulness(answer, retrieved, "")

    state["faithfulness"] = max(0.0, min(1.0, score))

    state["eval_retries"] = state.get("eval_retries", 0) + 1

    return state


def save_node(state):
    messages = state.get("messages", [])
    messages.append({"role": "assistant", "content": state["answer"]})
    state["messages"] = messages[-6:]
    return state


def route_decision(state: CapstoneState) -> str:
    return state.get("route", "skip")


def eval_decision(state: CapstoneState) -> str:
    if state.get("faithfulness", 1.0) < 0.7 and state.get("eval_retries", 0) < 2:
        return "answer"
    return "save"


graph = StateGraph(CapstoneState)

graph.add_node("memory", memory_node)
graph.add_node("router", router_node)
graph.add_node("retrieve", retrieval_node)
graph.add_node("skip", skip_retrieval_node)
graph.add_node("tool", tool_node)
graph.add_node("answer", answer_node)
graph.add_node("eval", eval_node)
graph.add_node("save", save_node)

graph.set_entry_point("memory")
graph.add_edge("memory", "router")
graph.add_conditional_edges(
    "router",
    route_decision,
    {
        "retrieve": "retrieve",
        "tool": "tool",
        "skip": "skip",
    },
)
graph.add_edge("retrieve", "answer")
graph.add_edge("skip", "answer")
graph.add_edge("tool", "answer")
graph.add_edge("answer", "eval")
graph.add_conditional_edges(
    "eval",
    eval_decision,
    {
        "answer": "answer",
        "save": "save",
    },
)
graph.add_edge("save", END)

app = graph.compile(checkpointer=MemorySaver())


def fresh_state(question: str) -> CapstoneState:
    return {
        "question": question,
        "messages": [],
        "route": "",
        "retrieved": "",
        "sources": [],
        "tool_result": "",
        "answer": "",
        "faithfulness": 0.0,
        "eval_retries": 0,
        "employee_name": "",
    }


def ask(question, thread_id):
    config = {"configurable": {"thread_id": thread_id}}

    result = app.invoke(
        {"question": question},
        config=config
    )

    return result


if __name__ == "__main__":
    print(f"KB size: {len(DOCUMENTS)}")
    for d in DOCUMENTS:
        print(f"- {d['id']}: {d['topic']}")

from typing import TypedDict, List

class CapstoneState(TypedDict):
    question: str
    messages: List[dict]
    route: str
    retrieved: str
    sources: List[str]
    tool_result: str
    answer: str
    faithfulness: float
    eval_retries: int

    # HR-specific field
    employee_name: str