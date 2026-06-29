# GenAI Learning Mentor

Mentor for personalized learning guidance.

## Problem
Students need personalized guidance to learn effectively and identify weak areas early.

## Architecture
```text
User
  ↓
RAG (Course Notes + Learning Resources)
  ↓
LLM (OpenAI/Gemini/Open-source via adapter)
  ↓
Response (Study Plan + Quiz + Weak Areas + Practice Questions)
```

### Core Components
- **Prompt Engineering**: Structured coaching prompts based on profile, weak areas, and context.
- **RAG Layer**: Retrieves relevant course notes/resources before generation.
- **Agent (Learning Coach)**: Produces adaptive tutoring output.
- **Outputs**:
  - Personalized study plan
  - Quiz generation
  - Weak-area identification
  - Bonus: Practice questions

## Prompt Design
Prompt template includes:
- Student profile and goals
- Weak areas identified from quiz scores
- Retrieved context from notes/resources
- Explicit tutoring instruction (next steps + checkpoint question)

## Evaluation
Minimal checks included in `tests/test_mentor.py`:
- Weak-area identification logic
- End-to-end coach output includes all required deliverables

Run:
```bash
python -m unittest discover -s tests
```

## Challenges
- Keeping implementation lightweight while still showing a full RAG + agent flow
- Supporting optional LLM dependencies without breaking local/test execution

## Future Enhancements
- Add FAISS/ChromaDB vector retrieval for semantic search
- Add Streamlit UI for student interaction
- Add richer quiz auto-evaluation and progress tracking
- Add Gemini/open-source model adapters

## Quick Start
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m genai_learning_mentor.app
```

Set `OPENAI_API_KEY` to use `OpenAILLM` in production workflows.
