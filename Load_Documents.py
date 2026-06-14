from pathlib import Path
import re

METADATA_FIELDS = {
    "Document ID": "document_id",
    "Owner": "owner",
    "Version": "version",
    "Effective date": "effective_date",
    "Applies to": "applies_to",
}


def extract_title(lines):
    for line in lines:
        if line.startswith("# "):
            return line.replace("# ", "", 1).strip()

    return "Untitled policy"


def extract_metadata(lines):
    metadata = {}

    for line in lines:
        match = re.match(r"^\*\*(.+?):\*\*\s*(.+?)\s*$", line.strip())

        if not match:
            continue

        field = match.group(1).strip()
        value = match.group(2).strip()

        if field in METADATA_FIELDS:
            metadata[METADATA_FIELDS[field]] = value

    return metadata

def load_markdown_documents(folder_path: str):
    folder = Path(folder_path)
    documents = []
    for file_path in folder.glob("*.md"):
        if file_path.name.lower() == "readme.md":
            continue
        text = file_path.read_text(encoding="utf-8")
        lines = text.splitlines()
        title  = extract_title(lines)
        metadata = extract_metadata(lines)

        document = {
            "text": text,
            "source": file_path.name,
            "title": title,
            "metadata": metadata
        }

        documents.append(document)

    return documents


if __name__ == "__main__":
    docs = load_markdown_documents("sample_docs")