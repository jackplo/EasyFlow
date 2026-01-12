# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

EasyFlow extends PocketFlow—a 100-line minimalist LLM framework—with optional quality-of-life utilities. The utilities are provider-agnostic routers for LLM calls and embeddings.

## Running Tests

```bash
# Run all tests
python3 -m pytest tests/

# Run a single test file
python3 -m pytest tests/test_utils_llm.py

# Run a specific test
python3 -m pytest tests/test_utils_llm.py::TestLLMBasicFunctionality::test_register_single_provider
```

## Architecture

### Core PocketFlow (pocketflow/**init**.py)

The framework models LLM workflows as a **Graph + Shared Store**:

- **Node**: Smallest building block with `prep(shared) -> exec(prep_res) -> post(shared, prep_res, exec_res)` lifecycle
  - `prep`: Read/preprocess data from shared store
  - `exec`: Execute compute logic (LLM calls, APIs). Should NOT access shared. Supports retries via `max_retries` and `wait` params
  - `post`: Write results to shared store, return action string to determine next node
- **Flow**: Orchestrates nodes using action-based transitions (`node_a >> node_b` or `node_a - "action" >> node_b`)
- **BatchNode/BatchFlow**: Process iterables, `exec` called per item
- **AsyncNode/AsyncFlow**: Async versions with `prep_async`, `exec_async`, `post_async`
- **AsyncParallelBatchNode/AsyncParallelBatchFlow**: Concurrent execution

### EasyFlow Utilities (pocketflow/utils/)

**LLM Provider Router** (`llm.py`):

```python
from pocketflow.utils import register_llm, call_llm

register_llm("openai", lambda prompt, model, **kw: ...)
call_llm("Hello!", "openai/gpt-4o")  # provider/model format
call_llm("Hello!", "gpt-4o")         # uses default (first registered)
```

**Embedding Router** (`embedding.py`):

```python
from pocketflow.utils import register_embedding, embed

register_embedding("openai", lambda text, model, **kw: ...)
embed("Hello world", "openai/text-embedding-3-small")
```

Both routers are thread-safe and support the `provider/model` format.

## Project Structure for PocketFlow Apps

When building a PocketFlow application, follow this structure:

```
my_project/
├── main.py          # Entry point, runs the flow
├── nodes.py         # Node class definitions
├── flow.py          # Flow construction and wiring
├── utils/           # Utility functions (one file per external API)
│   ├── call_llm.py
│   └── search_web.py
└── docs/
    └── design.md    # High-level design doc (no code)
```

## Key Patterns

**Node Pattern**: Separation of concerns between data access (prep/post) and compute (exec)

```python
class MyNode(Node):
    def prep(self, shared): return shared["input"]
    def exec(self, data): return call_llm(f"Process: {data}")
    def post(self, shared, prep_res, exec_res): shared["output"] = exec_res
```

**Avoid exception handling in exec()**: Let Node's built-in retry mechanism handle failures

**Structured Output**: Use YAML (not JSON) for LLM outputs—easier escaping for strings with quotes/newlines

## Agentic Coding Workflow

1. **Requirements** - Humans clarify requirements
2. **Flow Design** - Outline nodes and transitions (mermaid diagrams)
3. **Utilities** - Implement external API wrappers with simple tests
4. **Data Design** - Design the shared store schema
5. **Node Design** - Plan prep/exec/post for each node
6. **Implementation** - Keep it simple, fail fast, add logging
7. **Optimization** - Iterate on prompts and flow design
8. **Reliability** - Add retries, validation, self-evaluation nodes
