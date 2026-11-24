"""
Test critical fixes in src/chat_agent.py.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.chat_agent import ask_chat_agent, nl_to_intent
from src.schema import BuildingInfo, SceneJSON

def test_critical_queries():
    # Create dummy scene with 225 buildings: 142 undamaged in top-left, 83 damaged in bottom-right
    bldgs = []
    for i in range(142):
        bldgs.append(BuildingInfo(
            id=f"u_{i}",
            polygon=[[10, 10], [20, 10], [20, 20], [10, 20]],
            bbox=[10, 10, 20, 20],
            center=[15, 15],
            area_px=100.0,
            quadrant="top-left",
            halves=["top", "left"],
            damage="no-damage",
            confidence=1.0,
        ))
    for i in range(83):
        bldgs.append(BuildingInfo(
            id=f"d_{i}",
            polygon=[[600, 600], [610, 600], [610, 610], [600, 610]],
            bbox=[600, 600, 610, 610],
            center=[605, 605],
            area_px=100.0,
            quadrant="bottom-right",
            halves=["bottom", "right"],
            damage="destroyed",
            confidence=1.0,
        ))

    scene = SceneJSON(
        pair_id="test_scene",
        pre_image_path="pre.jpg",
        post_image_path="post.jpg",
        buildings=bldgs,
    )

    # 1. Test Query: "total number of buildings" -> must count all 225 buildings!
    res1 = ask_chat_agent("total number of buildings", scene)
    print("QUERY 1: 'total number of buildings'")
    print("  Intent:", res1["intent"])
    print("  Answer count:", res1["submission_json"]["answer"])
    assert res1["submission_json"]["answer"] == 225, f"Expected 225, got {res1['submission_json']['answer']}"
    assert res1["intent"]["damage_filter"] == "all"
    assert res1["intent"]["question_type"] == 1
    assert res1["intent"]["is_out_of_scope"] is False
    print("  Natural Response:\n" + res1["natural_response"])

    # 2. Test Query: "highlight the quadrant with most damaged buildings" -> must be Type 2, not Type 4!
    res2 = ask_chat_agent("highlight the quadrant with most damaged buildings", scene)
    print("\nQUERY 2: 'highlight the quadrant with most damaged buildings'")
    print("  Intent:", res2["intent"])
    print("  Type:", res2["intent"]["question_type"])
    print("  Is out of scope:", res2["intent"]["is_out_of_scope"])
    print("  Answer quadrant:", res2["submission_json"]["answer"])
    assert res2["intent"]["question_type"] == 2
    assert res2["intent"]["is_out_of_scope"] is False
    assert res2["submission_json"]["answer"] == "bottom-right"
    print("  Natural Response:\n" + res2["natural_response"])

    # 3. Test Query: "which area was hit hardest" -> must be Type 2, not Type 4!
    res3 = ask_chat_agent("which area was hit hardest", scene)
    print("\nQUERY 3: 'which area was hit hardest'")
    assert res3["intent"]["question_type"] == 2
    assert res3["intent"]["is_out_of_scope"] is False
    assert res3["submission_json"]["answer"] == "bottom-right"

    # 4. Test Query: Hospital (must remain Type 4 / out of scope)
    res4 = ask_chat_agent("which building is a hospital?", scene)
    print("\nQUERY 4: 'which building is a hospital?'")
    assert res4["intent"]["question_type"] == 4
    assert res4["intent"]["is_out_of_scope"] is True
    assert res4["submission_json"]["answer"] == "cannot be determined"

    print("\n>>> ALL CRITICAL DIAGNOSTIC CHECKS PASSED! <<<")

if __name__ == "__main__":
    test_critical_queries()
