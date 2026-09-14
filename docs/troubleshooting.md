# AARAG Troubleshooting Guide

## Table of Contents

- [Installation Issues](#installation-issues)
- [GPU/CUDA Issues](#gpucuda-issues)
- [Model Issues](#model-issues)
- [Performance Issues](#performance-issues)
- [Evaluation Issues](#evaluation-issues)
- [Demo Issues](#demo-issues)
- [Logging Issues](#logging-issues)

---

## Installation Issues

### ImportError: No module named 'xyz'

**Symptom:** Import error when running AARAG.

**Solution:**
```bash
# Reinstall all dependencies
pip install -r requirements.txt

# If using conda
conda install --file requirements.txt

# Check Python version (must be 3.10+)
python --version
```

### pip install fails with build errors

**Symptom:** Compilation errors during pip install.

**Solution:**
```bash
# Install build tools
# Ubuntu/Debian
sudo apt-get install build-essential python3-dev

# macOS
xcode-select --install

# Windows
# Install Visual Studio Build Tools
```

### Version conflicts

**Symptom:** Dependency version conflicts.

**Solution:**
```bash
# Create clean environment
python -m venv aarag-env-clean
source aarag-env-clean/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

## GPU/CUDA Issues

### CUDA out of memory

**Symptom:** `RuntimeError: CUDA out of memory`

**Solutions:**

1. **Enable 4-bit quantization:**
```yaml
# config/default.yaml
llm:
  load_in_4bit: true
```

2. **Reduce batch size:**
```yaml
training:
  router:
    batch_size: 16  # Reduce from 32
  evaluator:
    batch_size: 8   # Reduce from 16
```

3. **Use smaller model:**
```yaml
llm:
  model_name: "meta-llama/Llama-3.2-1B-Instruct"  # Smaller model
```

4. **Clear GPU cache:**
```python
import torch
torch.cuda.empty_cache()
```

### CUDA not available

**Symptom:** `torch.cuda.is_available()` returns False

**Solution:**
```bash
# Check NVIDIA driver
nvidia-smi

# Install CUDA toolkit
# https://developer.nvidia.com/cuda-downloads

# Reinstall PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Wrong CUDA version

**Symptom:** `CUDA error: no kernel image is available for execution on the device`

**Solution:**
```bash
# Check CUDA version
nvcc --version

# Install matching PyTorch
# CUDA 11.8
pip install torch --index-url https://download.pytorch.org/whl/cu118
# CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

---

## Model Issues

### Model not found

**Symptom:** `OSError: [model_name] not found`

**Solution:**
```bash
# Download models
python scripts/download_models.py

# Or download manually
from transformers import AutoModel, AutoTokenizer
AutoModel.from_pretrained("BAAI/bge-large-en-v1.5")
AutoTokenizer.from_pretrained("BAAI/bge-large-en-v1.5")
```

### HuggingFace authentication

**Symptom:** `401 Client Error: Unauthorized`

**Solution:**
```bash
# Login to HuggingFace
huggingface-cli login

# Or set token
export HUGGINGFACE_HUB_TOKEN="your_token_here"
```

### Slow model loading

**Symptom:** Models take a long time to load.

**Solutions:**

1. **Cache models locally:**
```bash
# Models are cached in ~/.cache/huggingface/
# Pre-download to avoid runtime downloads
python scripts/download_models.py
```

2. **Use model caching:**
```python
import os
os.environ["TRANSFORMERS_CACHE"] = "/path/to/cache"
```

---

## Performance Issues

### High latency

**Symptom:** Queries take too long to process.

**Diagnosis:**
```bash
# Analyze logs for bottlenecks
python src/utils/log_analyzer.py --log results/logs/aarag.log --report
```

**Solutions:**

1. **Enable query caching:**
```yaml
layers:
  query_cache:
    enabled: true
    max_size: 1000
    ttl_seconds: 3600
```

2. **Reduce retrieval scope:**
```yaml
knowledge_sources:
  vector_store:
    top_k: 3  # Reduce from 5
    chunk_size: 256  # Reduce from 512
```

3. **Disable expensive components:**
```python
response = aarag.query(
    "question",
    enable_web_fallback=False,  # Skip web search
    enable_self_reflection=False,  # Skip reflection
)
```

4. **Use GPU:**
```yaml
project:
  device: "cuda"
```

### Low accuracy

**Symptom:** Answers are inaccurate or irrelevant.

**Solutions:**

1. **Train models on domain data:**
```bash
python src/training/data_preparation.py
python src/training/train_router.py
python src/training/train_evaluator.py
```

2. **Increase retrieval scope:**
```yaml
knowledge_sources:
  vector_store:
    top_k: 10
```

3. **Enable all components:**
```python
response = aarag.query(
    "question",
    enable_web_fallback=True,
    enable_self_reflection=True,
)
```

### High memory usage

**Symptom:** System runs out of RAM.

**Solutions:**

1. **Use smaller embedding model:**
```yaml
knowledge_sources:
  vector_store:
    embedding_model: "BAAI/bge-small-en-v1.5"
```

2. **Reduce chunk size:**
```yaml
knowledge_sources:
  vector_store:
    chunk_size: 256
```

3. **Limit knowledge graph size:**
```yaml
knowledge_sources:
  knowledge_graph:
    max_hops: 2
```

---

## Evaluation Issues

### No evaluation data

**Symptom:** `No evaluation data found`

**Solution:**
```bash
# Download and prepare evaluation data
python src/training/data_preparation.py
```

### Tests failing

**Symptom:** pytest failures

**Solutions:**

1. **Run with verbose output:**
```bash
python -m pytest tests/ -v --tb=long
```

2. **Run specific test:**
```bash
python -m pytest tests/test_router.py -v
```

3. **Check Python version:**
```bash
python --version  # Must be 3.10+
```

### Evaluation metrics seem wrong

**Symptom:** Unexpected metric values

**Diagnosis:**
```bash
# Run with debug logging
python run_evals.py 2>&1 | tee eval_debug.log

# Check individual results
cat evals/results.json | python -m json.tool
```

---

## Demo Issues

### Streamlit won't start

**Symptom:** `streamlit: command not found`

**Solution:**
```bash
pip install streamlit
streamlit run demo/app.py
```

### Demo is slow

**Symptom:** Demo takes a long time to respond.

**Solutions:**

1. **Enable caching:**
```python
@st.cache_resource
def load_aarag():
    return AARAG(config_path="config/default.yaml")
```

2. **Use smaller model:**
```yaml
llm:
  model_name: ""  # Use fallback LLM
```

### Port already in use

**Symptom:** `Port 8501 is already in use`

**Solution:**
```bash
# Use different port
streamlit run demo/app.py --server.port 8502

# Or kill existing process
# Linux/Mac
lsof -ti:8501 | xargs kill -9
# Windows
netstat -ano | findstr :8501
taskkill /PID <PID> /F
```

---

## Logging Issues

### No log files created

**Symptom:** `results/logs/` directory is empty

**Solution:**
```bash
# Create log directory
mkdir -p results/logs

# Check permissions
ls -la results/logs/
```

### Logs are too verbose

**Symptom:** Log files are very large

**Solution:**
```yaml
# Adjust log level in code
from src.utils.logger import get_logger
logger = get_logger("aarag", level="INFO")  # Instead of DEBUG
```

### Structured logs not appearing

**Symptom:** JSON logs not generated

**Solution:**
```python
# Enable structured logging
from src.utils.structured_logger import get_structured_logger
logger = get_structured_logger("aarag", json_output=True, log_dir="logs/")
```

---

## Getting Help

If your issue is not covered here:

1. Check the [GitHub Issues](https://github.com/your-username/AARAG/issues)
2. Search the [Documentation](docs/)
3. Open a new issue with:
   - Python version
   - OS and architecture
   - GPU type (if relevant)
   - Steps to reproduce
   - Error messages/tracebacks
   - Relevant configuration
