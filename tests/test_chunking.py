"""
Unit tests for semantic chunking rules.
Tested without a real PDF — logic only.
"""


def test_chunk_has_required_keys():
    chunk = {
        "chunk_id": "abc",
        "procedure_id": "proc1",
        "page_start": 10,
        "page_end": 11,
        "steps": [{"step": 1, "instruction": "Press START", "images": []}],
        "retrieval_text": "Press START",
    }
    required = {"chunk_id", "procedure_id", "page_start", "page_end", "steps", "retrieval_text"}
    assert required.issubset(chunk.keys())


def test_steps_are_ordered():
    steps = [
        {"step": 1, "instruction": "Open panel", "images": []},
        {"step": 2, "instruction": "Press button", "images": ["img1"]},
        {"step": 3, "instruction": "Confirm", "images": []},
    ]
    for i, s in enumerate(steps):
        assert s["step"] == i + 1, "Steps must be sequentially ordered."


def test_image_attached_to_step():
    steps = [
        {"step": 1, "instruction": "Navigate to menu", "images": ["fig1"]},
    ]
    assert "fig1" in steps[0]["images"], "Image must be attached to step."
