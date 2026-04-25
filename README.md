# Building-Agentic-RAG-with-Llamaindex
# Agentic RAG with LlamaIndex
> A multi-phase project building increasingly intelligent RAG systems — from basic router query engines to full agentic document assistants that reason, retrieve, and summarize across multiple PDFs using LlamaIndex and GPT.

---

## What is Agentic RAG?

Standard RAG pipelines follow a fixed path: embed → retrieve → answer. **Agentic RAG** goes further — the LLM **actively decides** which retrieval strategy to use, which document to query, and whether to summarize or search, based on the question asked.

```
User Question
      ↓
[Agent] — reasons about the question
      ↓
Should I summarize or do vector search?
   ├── Summary question → SummaryIndex (tree_summarize)
   └── Specific question → VectorStoreIndex (similarity search + page filter)
      ↓
Final Answer
```

---

## Project Phases

This project is built across **4 progressive phases**, each adding more intelligence to the RAG pipeline.

---

### Phase 1 — Router Query Engine

The foundation. A single document (MetaGPT research paper) is loaded and two query engines are built — one for summarization, one for vector search. A **RouterQueryEngine** with `LLMSingleSelector` automatically routes each query to the right engine.

```
Query → LLMSingleSelector decides → SummaryIndex or VectorStoreIndex → Answer
```

**Key concepts:**
- `SimpleDirectoryReader` — loading PDF documents
- `SentenceSplitter` — chunking documents into nodes (chunk_size=1024)
- `SummaryIndex` — tree-based summarization over all nodes
- `VectorStoreIndex` — embedding-based similarity search
- `RouterQueryEngine` — intelligent routing between engines
- `LLMSingleSelector` — LLM decides which tool to use

---

### Phase 2 — Function Tools + Auto-Retrieval

Moving beyond fixed pipelines. Custom `FunctionTool` objects are defined and bound to the LLM directly using `predict_and_call()`. The LLM now decides **which tool to call and with what arguments** — including filtering by specific page numbers.

```python
# LLM automatically decides to call vector_tool with page filter
response = llm.predict_and_call(
    [vector_query_tool, summary_tool],
    "What are the MetaGPT comparisons with ChatDev on page 8?",
    verbose=True
)
```

**Key concepts:**
- `FunctionTool.from_defaults()` — wrapping Python functions as LLM-callable tools
- `MetadataFilters` + `FilterCondition.OR` — page-level filtering in vector search
- `predict_and_call()` — LLM selects and executes the right tool automatically
- `similarity_top_k=2` — retrieving top 2 most relevant chunks

---

### Phase 3 — Reusable Document Tool Factory

Abstracting everything into a clean, reusable function `get_doc_tools()` that takes any PDF file path and returns two ready-to-use tools — a vector search tool and a summary tool — both namespaced to the document.

```python
vector_tool, summary_tool = get_doc_tools(
    file_path="metagpt.pdf",
    name="metagpt"
)
# Returns: vector_tool_metagpt, summary_tool_metagpt
```

**Key concepts:**
- Factory function pattern for tool generation
- Dynamic tool naming with `f"vector_tool_{name}"`
- Optional page filtering — `page_numbers=None` by default
- Reusable across any PDF document

---

### Phase 4 — Multi-Document Agentic RAG

The most advanced phase. Multiple PDFs are loaded, each gets its own vector and summary tools via `get_doc_tools()`. An agent is built that can reason across all documents simultaneously — choosing which document to query and which strategy to use.

```
User Query
     ↓
[Agent] — has tools for each document
     ├── vector_tool_paper1 — specific questions about paper 1
     ├── summary_tool_paper1 — summarize paper 1
     ├── vector_tool_paper2 — specific questions about paper 2
     └── summary_tool_paper2 — summarize paper 2
     ↓
Reasoned Answer across multiple sources
```

**Key concepts:**
- Multi-document tool registry
- Agent reasoning across document boundaries
- Per-document namespaced tools
- `tree_summarize` response mode for long documents

---

## Architecture Overview

```
PDF Documents
      ↓
SimpleDirectoryReader → raw text
      ↓
SentenceSplitter → nodes (chunks of 1024 tokens)
      ↓
      ├── SummaryIndex → SummaryQueryEngine → summary_tool
      └── VectorStoreIndex → VectorQueryEngine → vector_tool
                                    ↓
                            MetadataFilters (page-level)
      ↓
LLM Agent (GPT-3.5-turbo / GPT-4)
      ↓
predict_and_call() → selects right tool → executes → returns answer
```

---

## Tech Stack

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square)
![LlamaIndex](https://img.shields.io/badge/LlamaIndex-Core-purple?style=flat-square)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--3.5%20%2F%20GPT--4-black?style=flat-square)
![RAG](https://img.shields.io/badge/RAG-Agentic-teal?style=flat-square)

- **LlamaIndex** — document loading, indexing, query engines, tools
- **OpenAI GPT-3.5-turbo / GPT-4** — LLM reasoning and tool selection
- **OpenAI text-embedding-ada-002** — document embeddings
- **SentenceSplitter** — intelligent document chunking
- **RouterQueryEngine** — automatic query routing
- **FunctionTool / QueryEngineTool** — LLM-callable tool wrappers

---

## Project Structure

```
Agentic-RAG-LlamaIndex/
│
├── phase1_router_query_engine.py    # Router between summary and vector search
├── phase2_function_tools.py         # Custom tools with auto-retrieval
├── phase3_doc_tool_factory.py       # Reusable get_doc_tools() function
├── phase4_multi_doc_agent.py        # Multi-document agentic RAG
├── utils.py                         # get_doc_tools() helper
├── metagpt.pdf                      # Sample research paper (test document)
├── requirements.txt                 # All dependencies
├── .env.example                     # API key template
└── README.md
```

---

## Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/Saqib00712/IBM_RAG_Specialization.git
cd IBM_RAG_Specialization
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set up API key
```bash
cp .env.example .env
```
Edit `.env`:
```
OPENAI_API_KEY=your_openai_api_key_here
```

### 4. Run any phase
```bash
python phase1_router_query_engine.py
# or open the notebooks in Jupyter
jupyter notebook
```

---

## Key Concepts Covered

- **Router Query Engine** — LLM automatically selects summary vs vector retrieval
- **FunctionTool** — converting Python functions into LLM-callable tools with docstrings as descriptions
- **Page-level metadata filtering** — querying specific pages of a PDF using `MetadataFilters`
- **predict_and_call()** — single LLM call that selects and executes the right tool
- **Factory pattern** — `get_doc_tools()` generates namespaced tools for any document
- **Multi-document reasoning** — agent queries across multiple PDFs simultaneously
- **tree_summarize** — recursive summarization mode for long documents
- **Async query engines** — `use_async=True` for faster parallel retrieval

---

## Example Queries

```python
# Phase 1 — Router automatically picks the right engine
query_engine.query("What is a summary of the MetaGPT paper?")
# → Routes to SummaryIndex

query_engine.query("What are the results on page 2?")
# → Routes to VectorStoreIndex with page filter

# Phase 2 — LLM picks the right tool
llm.predict_and_call(
    [vector_query_tool, summary_tool],
    "What are MetaGPT comparisons with ChatDev on page 8?",
    verbose=True
)

# Phase 3 — Reusable factory for any PDF
vector_tool, summary_tool = get_doc_tools("research_paper.pdf", "paper")

# Phase 4 — Multi-document agent
agent.query("Compare the methodologies across all papers")
```

---

## Related Certifications

Built as part of the IBM **RAG for Generative AI Applications Specialization** and **AI Agents Using RAG and LangChain** on Coursera.

[![IBM Badge](https://img.shields.io/badge/IBM-RAG%20Specialization-blue?style=flat-square)](https://www.credly.com/users/muhammad-saqib.361f9b8c)
[![IBM Badge](https://img.shields.io/badge/IBM-AI%20Agents%20Specialization-blue?style=flat-square)](https://www.credly.com/users/muhammad-saqib.361f9b8c)

---

## Author

**Muhammad Saqib**
- GitHub: [@Saqib00712](https://github.com/Saqib00712)
- LinkedIn: [muhammad-saqib](https://www.linkedin.com/in/muhammad-saqib-68b9b3374/)
- Email: saqibkhosa649@gmail.com
- Credly: [15x IBM Certified](https://www.credly.com/users/muhammad-saqib.361f9b8c)
