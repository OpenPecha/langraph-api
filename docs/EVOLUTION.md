# Project Evolution & Key Decisions

This document chronicles the development journey of the API, capturing the key architectural shifts, the problems encountered, and the reasoning behind the solutions. Understanding this evolution is key to understanding the final design.

### Phase 1: The Initial Monolithic Workflow

*   **Initial Concept**: The project began with a single, monolithic workflow. The idea was to have a single endpoint (`/translate`) that would perform translation and glossary extraction in one atomic operation.
*   **Implementation**: This was initially built using LangGraph to create a state machine: `Initialize -> Translate Batch -> Extract Glossary -> Finalize`.
*   **Problem Discovered**: While this worked for simple cases, it was incredibly rigid. The most significant issue was that glossary extraction was tightly coupled to the translation process. If a user just wanted to re-run the glossary extraction with a different prompt, they were forced to re-translate everything, wasting time and money.

### Research & Insight: The Need for Decoupling

*   **User Need**: Through iterative testing, it became clear that translation, glossary extraction, and standardization were distinct tasks that users wanted to control independently. A user might be happy with a translation but want to experiment with different glossary extraction settings.
*   **Technical Insight**: The "all-in-one" workflow created a data flow problem. Features like "Analyze Consistency" required the per-translation glossary data to be available, but other optimizations (like batching all glossary extractions at the end) meant this data wasn't ready in time. This conflict was the primary driver for the architectural refactor.

### Phase 2: The Great Refactoring - A Decoupled, Service-Oriented Architecture

The decision was made to break apart the monolithic workflow into a set of independent, stateless services.

1.  **Translation Service**: The `/translate` endpoints were simplified to do one thing well: translate text. All glossary logic was removed. This made them faster, simpler, and more predictable.

2.  **Glossary Service**: A new set of endpoints (`/glossary/extract`) was created.
    *   **Key Decision**: This service would take the *output* of the translation service as its *input*. This was the critical decoupling step.
    *   **Optimization Research**: We researched the most efficient way to handle potentially hundreds of glossary extractions. The initial "one-by-one" approach was too slow. The final implementation uses the `llm.abatch()` method, which sends a batch of prompts to the language model provider to be executed in parallel. This was further refined to send these requests in manageable chunks, preventing API errors.

3.  **Standardization Service**: This was built from the ground up with the decoupled mindset.
    *   **Analysis Endpoint**: A key requirement was a fast, non-LLM tool for finding inconsistencies. The solution uses pure Python `defaultdict` and `set` operations for a highly efficient single-pass analysis of the data.
    *   **Application Endpoint**: The core challenge was how to apply a standard without degrading the quality of the translation.
        *   **Initial Idea**: A simple find-and-replace. This was rejected as it would produce awkward, grammatically incorrect sentences.
        *   **Final Solution**: A sophisticated "minimal change" prompt was engineered. This prompt instructs the LLM to act as a linguistic expert, seamlessly integrating the standardized term while preserving the fluency of the original translation. This is a far superior approach.

### Phase 3: Streaming-First UI

Throughout the process, a heavy emphasis was placed on user experience.
*   **Initial State**: Only the translation process was streamed.
*   **Final State**: Every long-running, LLM-based operation (translation, glossary extraction, and standardization) has its own dedicated streaming endpoint. This provides a transparent and interactive experience for the user, allowing them to see results and progress in real-time for every stage of the workflow.

This evolutionary process of identifying problems, researching solutions, and refactoring led directly to the robust, flexible, and powerful architecture the project has today. 