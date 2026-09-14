"""Build Knowledge Graph — Construct KG from document corpus."""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.layers.knowledge_sources import DocumentLoader, KnowledgeGraphSearch


def build_kg(data_path: str, output_path: str = "data/knowledge_graph", chunk_size: int = 512):
    """Build knowledge graph from documents."""
    print(f"Building knowledge graph from: {data_path}")

    # Load documents
    loader = DocumentLoader(chunk_size=chunk_size)
    path = Path(data_path)

    if path.is_dir():
        documents = loader.load_directory(str(path))
    else:
        documents = loader.load_file(str(path))

    print(f"Loaded {len(documents)} documents")

    # Chunk documents
    chunks = loader.chunk_documents(documents)
    print(f"Created {len(chunks)} chunks")

    # Build knowledge graph
    kg = KnowledgeGraphSearch()
    kg_docs = [{"content": c.content, "doc_id": c.doc_id} for c in chunks]
    kg.build_from_documents(kg_docs)

    print(f"Knowledge graph built:")
    print(f"  Nodes: {len(kg.graph.nodes)}")
    print(f"  Edges: {len(kg.graph.edges)}")
    print(f"  Communities: {len(kg.communities)}")

    # Save
    kg.save(output_path)
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build knowledge graph from documents")
    parser.add_argument("--data", required=True, help="Path to data file or directory")
    parser.add_argument("--output", default="data/knowledge_graph", help="Output path")
    parser.add_argument("--chunk-size", type=int, default=512, help="Chunk size in tokens")
    args = parser.parse_args()

    build_kg(args.data, args.output, args.chunk_size)
