"""Download Models — Pre-download required models for AARAG."""

import os
import sys
from pathlib import Path

def download_models():
    """Download all required models."""
    print("=" * 60)
    print("AARAG Model Downloader")
    print("=" * 60)

    models = [
        ("sentence-transformers", "BAAI/bge-large-en-v1.5", "Embedding model"),
        ("transformers", "microsoft/deberta-v3-base", "Router classifier"),
        ("transformers", "microsoft/deberta-v3-small", "Retrieval evaluator"),
    ]

    for library, model_name, description in models:
        print(f"\nDownloading {description}: {model_name}")
        try:
            if library == "sentence-transformers":
                from sentence_transformers import SentenceTransformer
                SentenceTransformer(model_name)
            elif library == "transformers":
                from transformers import AutoTokenizer, AutoModel
                AutoTokenizer.from_pretrained(model_name)
                AutoModel.from_pretrained(model_name)
            print(f"  ✓ {model_name} downloaded successfully")
        except Exception as e:
            print(f"  ✗ Error downloading {model_name}: {e}")

    # Download NLTK data
    print("\nDownloading NLTK data...")
    try:
        import nltk
        nltk.download("punkt", quiet=True)
        nltk.download("stopwords", quiet=True)
        print("  ✓ NLTK data downloaded")
    except Exception as e:
        print(f"  ✗ NLTK error: {e}")

    print("\n" + "=" * 60)
    print("Model download complete!")
    print("=" * 60)


if __name__ == "__main__":
    download_models()
