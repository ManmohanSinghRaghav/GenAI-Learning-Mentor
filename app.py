import streamlit as st
from genai_learning_mentor.mentor import (
    InMemoryRAG,
    LearningMentor,
    GroqLLM,
    Resource,
)

st.set_page_config(page_title="GenAI Learning Mentor", page_icon="🎓")

resources = [
    Resource(
        title="Prompt Engineering Basics",
        content="Use role, task, constraints, and examples.",
    ),
    Resource(
        title="RAG Foundations",
        content="Retrieve relevant notes before generation to reduce hallucinations.",
    ),
    Resource(
        title="Adaptive Tutoring",
        content="Adjust difficulty based on quiz performance and weak areas.",
    ),
]

mentor = LearningMentor(
    rag=InMemoryRAG(resources),
    llm=GroqLLM(),
)

st.title("🎓 GenAI Learning Mentor")

query = st.text_input("Topic")
goal = st.text_input("Learning Goal", "Learn Prompt Engineering")
score = st.slider("Quiz Score", 0.0, 1.0, 0.5)

if st.button("Generate Guidance"):
    if not query:
        st.warning("Please enter a topic before generating guidance.")
    else:
        with st.spinner("Generating your personalized guidance..."):
            result = mentor.coach(
                query=query,
                student_profile={
                    "name": "Student",
                    "goals": [goal],
                },
                quiz_scores=[
                    {
                        "topic": query,
                        "score": score,
                    }
                ],
            )

        st.subheader("Tutor Response")
        st.write(result["response"])

        st.subheader("Study Plan")
        for step in result["personalized_study_plan"]:
            st.write("•", step)

        st.subheader("Weak Areas")
        weak = result["weak_area_identification"]
        st.write(", ".join(weak) if weak else "None identified")

        st.subheader("Practice Questions")
        for q in result["practice_questions"]:
            st.write("-", q)

        st.subheader("Quiz")
        for item in result["quiz_generation"]:
            st.write(item["question"])