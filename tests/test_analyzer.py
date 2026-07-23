import json

import pytest

from analyzer import build_analysis_prompt, parse_json_response


class DummyModel:
    def __init__(self, text):
        self.text = text

    def generate_content(self, prompt):
        return DummyModel(self.text)


@pytest.fixture
def sample_messages():
    return [
        {
            "id": 1,
            "date": "27/03/26",
            "time": "8:53 am",
            "sender": "Career Counselor",
            "text": "Hello there",
        }
    ]


def test_build_analysis_prompt_includes_messages(sample_messages):
    prompt = build_analysis_prompt(sample_messages)

    assert "Message ID: 1" in prompt
    assert "Career Counselor" in prompt
    assert "Hello there" in prompt


def test_parse_json_response_valid_json():
    text = json.dumps({"overall_score": 90})
    parsed = parse_json_response(text)

    assert parsed["overall_score"] == 90


def test_parse_json_response_invalid_json():
    parsed = parse_json_response("not json")

    assert "error" in parsed
    assert parsed["raw_response"] == "not json"