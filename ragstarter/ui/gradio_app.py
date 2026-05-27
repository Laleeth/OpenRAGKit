from __future__ import annotations

from pathlib import Path

import gradio as gr

from ragstarter.api.deps import get_service_container
from ragstarter.services.container import ingest_loaded_document
from ragstarter.services.text_loading import load_file


def _resolve_file_path(file_obj) -> Path | None:
    if file_obj is None:
        return None
    if isinstance(file_obj, (str, Path)):
        return Path(file_obj)
    for attr in ("path", "name"):
        value = getattr(file_obj, attr, None)
        if value:
            return Path(value)
    return None


def index_file(file_obj):
    services = get_service_container()
    path = _resolve_file_path(file_obj)
    if path is None:
        return "No file selected.", []
    loaded = load_file(path)
    record, chunks = ingest_loaded_document(services, loaded)
    return f"Indexed {record.source_name} with {len(chunks)} chunks.", [record.to_dict()]


def do_search(query, top_k):
    services = get_service_container()
    hits = services.retriever.search(query, top_k=top_k or services.settings.top_k)
    rows = []
    for hit in hits:
        rows.append(
            {
                "score": round(hit.score, 4),
                "source": hit.source_name,
                "page": f"{hit.page_start}-{hit.page_end}" if hit.page_start else "",
                "content": hit.content[:300],
            }
        )
    return rows


def do_chat(question, top_k):
    services = get_service_container()
    result = services.rag.answer(question, top_k=top_k or services.settings.top_k)
    sources = []
    for i, hit in enumerate(result.sources, start=1):
        sources.append(f"[{i}] {hit.source_name} ({hit.page_start}-{hit.page_end})\n{hit.content[:400]}")
    return result.answer, "\n\n".join(sources), result.provider


def build_demo():
    with gr.Blocks(title="RAG Starter Kit") as demo:
        gr.Markdown("# RAG Starter Kit")
        gr.Markdown("Upload docs, inspect retrieval, and chat over your private corpus.")
        with gr.Tab("Index"):
            file_input = gr.File(label="Upload document", file_count="single", type="filepath")
            index_btn = gr.Button("Index document")
            index_status = gr.Markdown()
            indexed_docs = gr.JSON()
            index_btn.click(index_file, inputs=[file_input], outputs=[index_status, indexed_docs])
        with gr.Tab("Search"):
            q = gr.Textbox(label="Query")
            k = gr.Slider(1, 20, value=5, step=1, label="Top K")
            search_btn = gr.Button("Search")
            search_output = gr.JSON()
            search_btn.click(do_search, inputs=[q, k], outputs=[search_output])
        with gr.Tab("Chat"):
            question = gr.Textbox(label="Question")
            k2 = gr.Slider(1, 20, value=5, step=1, label="Top K")
            chat_btn = gr.Button("Answer")
            answer = gr.Textbox(label="Answer", lines=10)
            sources = gr.Textbox(label="Sources", lines=10)
            provider = gr.Textbox(label="Provider")
            chat_btn.click(do_chat, inputs=[question, k2], outputs=[answer, sources, provider])
    return demo


if __name__ == "__main__":
    demo = build_demo()
    demo.launch(server_name="0.0.0.0", server_port=7860)
