"""Helpers — Common utility functions for AARAG."""

import os
import json
import time
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional
from functools import wraps


def timer(func):
    """Decorator to time function execution."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        print(f"{func.__name__} took {elapsed:.3f}s")
        return result
    return wrapper


def load_yaml(path: str) -> dict:
    """Load YAML configuration file."""
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_json(data: Any, path: str, indent: int = 2):
    """Save data to JSON file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, default=str)


def load_json(path: str) -> Any:
    """Load data from JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def truncate_text(text: str, max_length: int = 500, suffix: str = "...") -> str:
    """Truncate text to max_length with suffix."""
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def hash_text(text: str) -> str:
    """Generate a short hash for text."""
    return hashlib.md5(text.encode()).hexdigest()[:12]


def ensure_dir(path: str) -> Path:
    """Ensure directory exists and return Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_device() -> str:
    """Get the best available device."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def format_latency(seconds: float) -> str:
    """Format latency in human-readable form."""
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    else:
        return f"{seconds / 60:.1f}m"


def batch_items(items: List[Any], batch_size: int) -> List[List[Any]]:
    """Split items into batches."""
    return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]
