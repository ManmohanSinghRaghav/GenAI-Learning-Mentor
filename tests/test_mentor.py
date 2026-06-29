import unittest

from genai_learning_mentor.mentor import InMemoryRAG, LearningMentor, Resource


class TestLearningMentor(unittest.TestCase):
    def setUp(self) -> None:
        resources = [
            Resource("Prompt Engineering", "Role prompting and few-shot examples."),
            Resource("RAG Notes", "Retrieval augmented generation uses relevant context."),
        ]
        self.mentor = LearningMentor(InMemoryRAG(resources))

    def test_weak_area_identification(self):
        weak = self.mentor.identify_weak_areas([
            {"topic": "Prompt Engineering", "score": 0.6},
            {"topic": "RAG", "score": 0.9},
        ])
        self.assertEqual(weak, ["Prompt Engineering"])

    def test_coach_outputs_required_fields(self):
        result = self.mentor.coach(
            query="prompt engineering",
            student_profile={"name": "A", "goals": ["improve prompts"]},
            quiz_scores=[{"topic": "prompt engineering", "score": 0.5}],
        )
        self.assertIn("personalized_study_plan", result)
        self.assertIn("quiz_generation", result)
        self.assertIn("weak_area_identification", result)
        self.assertIn("practice_questions", result)
        self.assertTrue(result["retrieved_context"])


if __name__ == "__main__":
    unittest.main()
