from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

try:
    from langchain_core.prompts import PromptTemplate
except Exception:  # pragma: no cover
    try:
        from langchain.prompts import PromptTemplate  # type: ignore
    except Exception:  # pragma: no cover
        PromptTemplate = None  # type: ignore

try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore


@dataclass
class Resource:
    title: str
    content: str


class InMemoryRAG:
    """Simple RAG retriever backed by in-memory course notes/resources."""

    def __init__(self, resources: List[Resource]):
        self.resources = resources

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        return set(re.findall(r"[a-zA-Z0-9]+", text.lower()))

    def retrieve(self, query: str, k: int = 3) -> List[Resource]:
        query_tokens = self._tokenize(query)
        scored: List[tuple[int, Resource]] = []
        for resource in self.resources:
            score = len(query_tokens & self._tokenize(resource.content + " " + resource.title))
            if score > 0:
                scored.append((score, resource))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [resource for _, resource in scored[:k]]


class OpenAILLM:
    """OpenAI wrapper compatible with the mentor's generation interface."""

    def __init__(self, model: str = "gpt-4o-mini", api_key: Optional[str] = None):
        if OpenAI is None:
            raise RuntimeError("openai package is not installed.")
        self.model = model
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

    def generate(self, prompt: str) -> str:
        response = self.client.responses.create(
            model=self.model,
            input=prompt,
            temperature=0.3,
        )
        return response.output_text.strip()


class LearningMentor:
    """Adaptive tutoring agent with prompt engineering + RAG context."""

    def __init__(self, rag: InMemoryRAG, llm: Optional[Any] = None):
        self.rag = rag
        self.llm = llm

    def _build_prompt(self, student_profile: Dict[str, Any], query: str, context: str) -> str:
        template = (
            "You are a learning coach.\\n"
            "Student profile: {student_profile}\\n"
            "Weak areas: {weak_areas}\\n"
            "Question: {query}\\n"
            "Retrieved context: {context}\\n"
            "Provide clear next steps, examples, and one checkpoint question."
        )
        weak_areas = ", ".join(student_profile.get("weak_areas", [])) or "None identified"
        if PromptTemplate is None:
            return template.format(
                student_profile=student_profile,
                weak_areas=weak_areas,
                query=query,
                context=context,
            )
        prompt = PromptTemplate.from_template(template)
        return prompt.format(
            student_profile=student_profile,
            weak_areas=weak_areas,
            query=query,
            context=context,
        )

    def identify_weak_areas(self, quiz_scores: List[Dict[str, Any]], threshold: float = 0.7) -> List[str]:
        weak_areas = []
        for result in quiz_scores:
            if float(result.get("score", 0)) < threshold:
                weak_areas.append(str(result.get("topic", "unknown")))
        return sorted(set(weak_areas))

    def personalized_study_plan(self, student_profile: Dict[str, Any], weak_areas: List[str]) -> List[str]:
        goals = student_profile.get("goals", ["master course outcomes"])
        plan = [
            f"Day 1-2: Review fundamentals for {', '.join(weak_areas) if weak_areas else 'current module'}.",
            f"Day 3-4: Apply concepts through practice aligned to goal: {goals[0]}.",
            "Day 5: Take a quiz and reflect on mistakes.",
            "Day 6-7: Revise weak concepts and attempt advanced problems.",
        ]
        return plan

    def generate_quiz(self, topic: str, n_questions: int = 3) -> List[Dict[str, str]]:
        quiz = []
        for i in range(1, n_questions + 1):
            quiz.append(
                {
                    "question": f"Q{i}. Explain a core idea in {topic} and provide one example.",
                    "answer": f"Core idea and example in {topic}.",
                }
            )
        return quiz

    def generate_practice_questions(self, topic: str, n_questions: int = 5) -> List[str]:
        return [f"Practice {i}: Solve a {topic} problem with step-by-step reasoning." for i in range(1, n_questions + 1)]

    def coach(self, query: str, student_profile: Dict[str, Any], quiz_scores: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        quiz_scores = quiz_scores or []
        weak_areas = self.identify_weak_areas(quiz_scores)
        student_profile = {**student_profile, "weak_areas": weak_areas}
        retrieved = self.rag.retrieve(query)
        context = "\n".join(f"- {doc.title}: {doc.content}" for doc in retrieved) or "No matching notes found."
        prompt = self._build_prompt(student_profile, query, context)

        if self.llm is not None:
            response = self.llm.generate(prompt)
        else:
            response = (
                "Adaptive Tutor Response:\n"
                f"Focus today on: {', '.join(weak_areas) if weak_areas else 'current topic progress'}.\n"
                "Use the study plan and checkpoint yourself with one summary question."
            )

        return {
            "response": response,
            "personalized_study_plan": self.personalized_study_plan(student_profile, weak_areas),
            "quiz_generation": self.generate_quiz(query, n_questions=3),
            "weak_area_identification": weak_areas,
            "practice_questions": self.generate_practice_questions(query, n_questions=5),
            "prompt": prompt,
            "retrieved_context": context,
        }
