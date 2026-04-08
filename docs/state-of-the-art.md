# State of the Art

## Problem Framing

Large language models are effective on mainstream programming languages but significantly less reliable on proprietary DSLs.
In such settings, model outputs can be fluent yet syntactically or semantically invalid.

## Baseline Approaches

- **Prompt-only approach:** low cost, but unreliable for unseen DSL conventions.
- **Fine-tuning:** potentially strong but expensive, slower to update, and operationally rigid.
- **RAG approach:** adaptive and easier to maintain when source code and documentation evolve.

## Limits of Naive RAG

Pure vector retrieval is effective for conceptual proximity but weak for exact symbol-level lookup.
For DSL engineering tasks, exact lexical constraints are often mandatory.

## Agentic Direction

Recent agentic paradigms improve robustness by combining:

- tool selection;
- iterative planning and backtracking;
- evidence-driven synthesis before final answer emission.

This motivates the hybrid and agentic architecture adopted in Envision.
