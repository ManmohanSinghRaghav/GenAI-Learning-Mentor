from genai_learning_mentor.mentor import InMemoryRAG, LearningMentor, Resource


def create_default_mentor() -> LearningMentor:
    resources = [
        Resource(title="Prompt Engineering Basics", content="Use role, task, constraints, and examples."),
        Resource(title="RAG Foundations", content="Retrieve relevant notes before generation to reduce hallucinations."),
        Resource(title="Adaptive Tutoring", content="Adjust difficulty based on quiz performance and weak areas."),
    ]
    rag = InMemoryRAG(resources)
    return LearningMentor(rag=rag)


def run_demo() -> None:
    mentor = create_default_mentor()
    profile = {"name": "Student", "goals": ["learn prompt engineering"]}
    output = mentor.coach(
        query="prompt engineering",
        student_profile=profile,
        quiz_scores=[{"topic": "prompt design", "score": 0.5}, {"topic": "RAG", "score": 0.8}],
    )
    print(output["response"])
    print("\nStudy Plan:")
    for step in output["personalized_study_plan"]:
        print(f"- {step}")


if __name__ == "__main__":
    run_demo()
