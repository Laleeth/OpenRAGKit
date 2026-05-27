from __future__ import annotations

from pathlib import Path

from ragstarter.api.deps import get_service_container
from ragstarter.services.text_loading import load_text
from ragstarter.services.container import ingest_loaded_document


def main():
    services = get_service_container()
    examples = Path("examples")
    examples.mkdir(parents=True, exist_ok=True)

    sample1 = examples / "sample_policy.md"
    sample1.write_text(
        "# Refund Policy\n\nYou can request a refund within 7 days of purchase if the product has not been used.\n\nSupport hours are Monday to Friday, 10 AM to 6 PM.",
        encoding="utf-8",
    )

    sample2 = examples / "sample_faq.txt"
    sample2.write_text(
        "How do I reset my password? Use the account settings page and click Reset Password.\n\nHow long does indexing take? Usually under a minute for short docs.",
        encoding="utf-8",
    )

    for path in [sample1, sample2]:
        doc = load_text(path.read_text(encoding="utf-8"), source_name=path.name, source_type="markdown", source_uri=str(path), title=path.stem)
        record, chunks = ingest_loaded_document(services, doc)
        print(f"Seeded {record.source_name}: {len(chunks)} chunks")


if __name__ == "__main__":
    main()
