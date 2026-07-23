import pytest

from parser import parse_chat


def test_parse_chat_single_message():
    text = "27/03/26, 8:53 am - Career Counselor: Hello there"
    messages = parse_chat(text)

    assert len(messages) == 1
    assert messages[0]["id"] == 1
    assert messages[0]["date"] == "27/03/26"
    assert messages[0]["time"] == "8:53 am"
    assert messages[0]["sender"] == "Career Counselor"
    assert messages[0]["text"] == "Hello there"


def test_parse_chat_multiline_message():
    text = (
        "27/03/26, 8:53 am - Career Counselor: Hello there\n"
        "This is a multiline message.\n"
        "27/03/26, 9:00 am - Parent: Hi"
    )
    messages = parse_chat(text)

    assert len(messages) == 2
    assert "multiline message" in messages[0]["text"]
    assert messages[1]["sender"] == "Parent"