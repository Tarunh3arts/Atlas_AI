"""
Unit and integration tests for src/chat_agent.py.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.chat_agent import ask_chat_agent, nl_to_intent
from src.data_utils import gt_label_to_scene_json
from src.schema import AnswerResponse

def test_chat_pipeline():
    lbl_p = Path("Dataset/CIT_Hackathon_PS04_aerial/practice/labels/p11b4ad02.json")
    pre_p = Path("Dataset/CIT_Hackathon_PS04_aerial/practice/images/p11b4ad02_pre.jpg")
    post_p = Path("Dataset/CIT_Hackathon_PS04_aerial/practice/images/p11b4ad02_post.jpg")

    scene = gt_label_to_scene_json(lbl_p, "p11b4ad02", str(pre_p), str(post_p))

    test_cases = [
        ("how many houses collapsed in the northern half?", 1),
        ("which quadrant had the worst destruction?", 2),
        ("is the top half more damaged than the bottom half?", 3),
        ("who owns the destroyed property on the right?", 4),
        ("what is the total repair cost for this disaster?", 4),
        ("is there a hospital visible in the bottom-left?", 4),
        ("calculate the ratio of undamaged buildings to damaged buildings", 1),
    ]

    for question, expected_type in test_cases:
        res = ask_chat_agent(question, scene)
        ans_obj = res["answer_response"]
        sub = res["submission_json"]

        assert "id" in sub and "answer" in sub and "region" in sub
        assert ans_obj.question_type == expected_type, f"Expected Type {expected_type}, got {ans_obj.question_type} for '{question}'"
        print(f"PASS: '{question}' -> Type {ans_obj.question_type} | Answer: {sub['answer']} | Backend: {res['backend']}")

if __name__ == "__main__":
    test_chat_pipeline()
    print("\nAll chat agent tests passed successfully!")
