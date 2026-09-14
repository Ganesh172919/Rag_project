"""Document Loader — Ingest and chunk documents for vector store and KG."""

import os
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from tqdm import tqdm


@dataclass
class Document:
    """Represents a single document or chunk."""
    content: str
    metadata: dict = field(default_factory=dict)
    doc_id: str = ""

    def __post_init__(self):
        if not self.doc_id:
            self.doc_id = hashlib.md5(self.content.encode()).hexdigest()[:12]


@dataclass
class Chunk:
    """A chunk of a document."""
    content: str
    doc_id: str
    chunk_index: int
    metadata: dict = field(default_factory=dict)
    chunk_id: str = ""

    def __post_init__(self):
        if not self.chunk_id:
            self.chunk_id = f"{self.doc_id}_{self.chunk_index}"


class DocumentLoader:
    """Load documents from files and chunk them for retrieval."""

    SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf", ".json", ".jsonl", ".csv"}

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def load_file(self, file_path: str) -> List[Document]:
        """Load a single file into documents."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        ext = path.suffix.lower()
        if ext not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {ext}")

        if ext in {".txt", ".md"}:
            return self._load_text(path)
        elif ext == ".pdf":
            return self._load_pdf(path)
        elif ext == ".json":
            return self._load_json(path)
        elif ext == ".jsonl":
            return self._load_jsonl(path)
        elif ext == ".csv":
            return self._load_csv(path)
        return []

    def load_directory(self, dir_path: str, recursive: bool = True) -> List[Document]:
        """Load all supported files from a directory."""
        path = Path(dir_path)
        if not path.is_dir():
            raise NotADirectoryError(f"Not a directory: {dir_path}")

        documents = []
        pattern = "**/*" if recursive else "*"
        for file_path in sorted(path.glob(pattern)):
            if file_path.is_file() and file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                try:
                    documents.extend(self.load_file(str(file_path)))
                except Exception as e:
                    print(f"Warning: Failed to load {file_path}: {e}")
        return documents

    def chunk_documents(self, documents: List[Document]) -> List[Chunk]:
        """Split documents into chunks with overlap."""
        all_chunks = []
        for doc in tqdm(documents, desc="Chunking documents"):
            chunks = self._chunk_text(doc.content)
            for i, chunk_text in enumerate(chunks):
                all_chunks.append(Chunk(
                    content=chunk_text,
                    doc_id=doc.doc_id,
                    chunk_index=i,
                    metadata={**doc.metadata, "source": doc.metadata.get("source", "")},
                ))
        return all_chunks

    def _chunk_text(self, text: str) -> List[str]:
        """Split text into chunks of chunk_size with chunk_overlap."""
        if len(text) <= self.chunk_size:
            return [text]

        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]

            # Try to break at sentence boundary
            if end < len(text):
                last_period = chunk.rfind(".")
                last_newline = chunk.rfind("\n")
                break_point = max(last_period, last_newline)
                if break_point > self.chunk_size * 0.3:
                    chunk = chunk[:break_point + 1]
                    end = start + break_point + 1

            chunks.append(chunk.strip())
            start = end - self.chunk_overlap

        return [c for c in chunks if c]

    def _load_text(self, path: Path) -> List[Document]:
        content = path.read_text(encoding="utf-8", errors="ignore")
        return [Document(content=content, metadata={"source": str(path), "type": "text"})]

    def _load_pdf(self, path: Path) -> List[Document]:
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(str(path))
            pages = []
            for i, page in enumerate(doc):
                text = page.get_text()
                if text.strip():
                    pages.append(Document(
                        content=text,
                        metadata={"source": str(path), "type": "pdf", "page": i + 1},
                    ))
            return pages
        except ImportError:
            # Fallback: try pdfplumber
            try:
                import pdfplumber
                with pdfplumber.open(str(path)) as pdf:
                    pages = []
                    for i, page in enumerate(pdf.pages):
                        text = page.extract_text() or ""
                        if text.strip():
                            pages.append(Document(
                                content=text,
                                metadata={"source": str(path), "type": "pdf", "page": i + 1},
                            ))
                    return pages
            except ImportError:
                raise ImportError("Install PyMuPDF or pdfplumber for PDF support: pip install PyMuPDF")

    def _load_json(self, path: Path) -> List[Document]:
        import json
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return [Document(
                content=str(item),
                metadata={"source": str(path), "type": "json", "index": i},
            ) for i, item in enumerate(data)]
        return [Document(content=str(data), metadata={"source": str(path), "type": "json"})]

    def _load_jsonl(self, path: Path) -> List[Document]:
        import json
        docs = []
        for i, line in enumerate(path.read_text(encoding="utf-8").strip().split("\n")):
            if line.strip():
                data = json.loads(line)
                docs.append(Document(
                    content=data.get("text", data.get("content", str(data))),
                    metadata={"source": str(path), "type": "jsonl", "index": i},
                ))
        return docs

    def _load_csv(self, path: Path) -> List[Document]:
        import csv
        docs = []
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                content = " | ".join(f"{k}: {v}" for k, v in row.items() if v)
                docs.append(Document(
                    content=content,
                    metadata={"source": str(path), "type": "csv", "index": i},
                ))
        return docs
