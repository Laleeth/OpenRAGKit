from ragstarter.services.chunking import Chunker


def test_chunker_creates_chunks():
    chunker = Chunker(chunk_size=20, chunk_overlap=5)
    text = "## Title\n\nThis is a sentence. " * 20
    chunks = chunker.chunk_text(
        text=text,
        document_id="doc1",
        source_name="sample.md",
        source_type="markdown",
        title="Sample",
    )
    assert chunks
    assert chunks[0].chunk_id == "doc1:0"
    assert chunks[0].document_id == "doc1"
    assert chunks[0].content
