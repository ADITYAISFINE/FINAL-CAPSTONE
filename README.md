# FINAL-CAPSTONE
# HR Policy Assistant (Agentic AI Capstone Project)

## Project Overview

HR Policy Assistant is an AI-powered assistant built using LangGraph, ChromaDB, and Streamlit. It helps employees get quick and accurate answers for HR-related queries such as leave policy, sick leave, working hours, payroll, resignation process, and employee benefits.

The system uses Retrieval-Augmented Generation (RAG), memory, tool usage, and self-evaluation to provide grounded answers while avoiding hallucination.

---

## Features

- LangGraph-based multi-node workflow
- ChromaDB knowledge base with 10 HR policy documents
- Memory support using `thread_id`
- Employee name recall across conversations
- Tool support for:
  - Calculator
  - Current date and time
- Self-reflection using evaluation node
- Streamlit UI for interactive chat
- Red-team safety handling:
  - Out-of-scope refusal
  - Prompt injection protection

---

## System Flow

User → Memory Node → Router → (Retrieve / Tool / Skip) → Answer → Eval → Save → END

### Nodes

- **Memory Node** → extracts employee name + manages memory
- **Router Node** → decides retrieve / tool / skip
- **Retrieval Node** → fetches HR policy data from ChromaDB
- **Tool Node** → handles calculator + date/time queries
- **Answer Node** → generates grounded response
- **Eval Node** → checks answer faithfulness
- **Save Node** → stores final response

---

## Knowledge Base Topics

1. Leave Policy
2. Sick Leave
3. Working Hours
4. Attendance
5. Payroll
6. Overtime
7. Holidays
8. Code of Conduct
9. Resignation
10. Employee Benefits

---

## Tech Stack

- Python
- LangGraph
- LangChain
- ChromaDB
- Sentence Transformers
- Streamlit
- Pandas
- Regex
- Python Dotenv
- Groq API

---

## Project Structure

```text
project/
│
├── agent.py
├── capstone_streamlit.py
├── day13_capstone.ipynb
├── requirements.txt
├── .env
├── .gitignore
│
└── knowledge_base/
    └── hr_docs.py
