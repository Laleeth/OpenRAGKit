from pathlib import Path

from ragstarter.services.text_loading import load_text


def test_load_text_hashes_content():
    doc = load_text("hello world", source_name="a.txt", source_type="text")
    assert doc.content_hash
    assert doc.text_length == len("hello world")
