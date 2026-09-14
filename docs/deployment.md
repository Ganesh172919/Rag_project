# AARAG Deployment Guide

## 1. Overview

This guide covers deploying AARAG in various environments, from local development to production cloud deployments.

---

## 2. Local Development

### 2.1 Quick Start

```bash
# Clone and setup
cd ~/AARAG
python -m venv aarag-env
source aarag-env/bin/activate
pip install -r requirements.txt

# Run tests
python -m pytest tests/ -v

# Run pipeline
python run_pipeline.py --query "What is machine learning?"

# Launch demo
streamlit run demo/app.py
```

### 2.2 Configuration

Edit `config/default.yaml`:

```yaml
# For CPU-only deployment
project:
  device: "cpu"

llm:
  model_name: ""  # Use fallback LLM (no GPU required)
```

---

## 3. Docker Deployment

### 3.1 Dockerfile

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Create necessary directories
RUN mkdir -p data/raw data/processed data/knowledge_graph \
    results/logs results/figures results/tables \
    evals/figures evals/logs logs

# Expose port for demo
EXPOSE 8501

# Default command
CMD ["python", "run_pipeline.py", "--query", "What is machine learning?"]
```

### 3.2 Build and Run

```bash
# Build image
docker build -t aarag .

# Run with GPU
docker run --gpus all -p 8501:8501 aarag streamlit run demo/app.py --server.port 8501

# Run evaluation
docker run --gpus all aarag python run_evals.py

# Run with custom config
docker run -v $(pwd)/config:/app/config aarag python run_pipeline.py --config config/custom.yaml
```

### 3.3 Docker Compose

```yaml
version: '3.8'

services:
  aarag:
    build: .
    ports:
      - "8501:8501"
    volumes:
      - ./data:/app/data
      - ./models:/app/models
      - ./config:/app/config
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    command: streamlit run demo/app.py --server.port 8501 --server.address 0.0.0.0

  # Optional: Neo4j for production knowledge graph
  neo4j:
    image: neo4j:5.0
    ports:
      - "7474:7474"
      - "7687:7687"
    environment:
      NEO4J_AUTH: neo4j/password
    volumes:
      - neo4j_data:/data

volumes:
  neo4j_data:
```

---

## 4. Cloud Deployment

### 4.1 AWS (EC2 with GPU)

```bash
# Launch EC2 instance (g4dn.xlarge for T4 GPU)
# Install NVIDIA drivers and Docker

# Pull and run
docker pull aarag:latest
docker run --gpus all -p 8501:8501 aarag:latest
```

**Recommended instances:**
| Instance | GPU | VRAM | Cost/hr |
|---|---|---|---|
| g4dn.xlarge | T4 | 16GB | $0.526 |
| g5.xlarge | A10G | 24GB | $1.006 |
| p3.2xlarge | V100 | 16GB | $3.06 |

### 4.2 Google Cloud (GKE)

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: aarag
spec:
  replicas: 1
  selector:
    matchLabels:
      app: aarag
  template:
    metadata:
      labels:
        app: aarag
    spec:
      containers:
      - name: aarag
        image: gcr.io/project-id/aarag:latest
        ports:
        - containerPort: 8501
        resources:
          limits:
            nvidia.com/gpu: 1
          requests:
            memory: "16Gi"
            cpu: "4"
```

### 4.3 Azure (AKS)

```bash
# Create GPU node pool
az aks nodepool add \
    --resource-group aarag-rg \
    --cluster-name aarag-cluster \
    --name gpupool \
    --node-count 1 \
    --node-vm-size Standard_NC6s_v3 \
    --enable-cluster-autoscaler \
    --min-count 1 \
    --max-count 3
```

---

## 5. Production Considerations

### 5.1 Scaling

**Horizontal Scaling:**
- Deploy multiple AARAG instances behind a load balancer
- Use Redis for shared query cache across instances
- Use external vector store (Pinecone, Weaviate) for shared knowledge base

**Vertical Scaling:**
- Increase GPU memory for larger models
- Use model parallelism for very large models
- Enable 4-bit quantization to reduce memory usage

### 5.2 Monitoring

**Health Check Endpoint:**
```python
# Add to demo/app.py or create FastAPI wrapper
@app.route("/health")
def health():
    return {"status": "healthy", "version": "1.1.0"}
```

**Metrics to Monitor:**
- Query latency (p50, p95, p99)
- Error rate
- GPU utilization
- Memory usage
- Cache hit rate

### 5.3 Security

**Network Security:**
- Use HTTPS for all endpoints
- Implement API key authentication
- Rate limiting to prevent abuse
- Input validation and sanitization

**Data Security:**
- Encrypt data at rest
- Encrypt data in transit (TLS)
- Regular security audits
- Dependency vulnerability scanning

### 5.4 Backup and Recovery

**What to Backup:**
- Vector store index files
- Knowledge graph data
- Trained model checkpoints
- Configuration files
- Evaluation results

**Backup Strategy:**
```bash
# Daily backup script
tar -czf backup_$(date +%Y%m%d).tar.gz \
    models/ \
    data/knowledge_graph/ \
    config/ \
    evals/
```

---

## 6. Performance Tuning

### 6.1 GPU Optimization

```yaml
# config/production.yaml
llm:
  load_in_4bit: true  # Reduce memory by 75%
  max_new_tokens: 512  # Limit generation length
  temperature: 0.1     # Deterministic generation

knowledge_sources:
  vector_store:
    backend: "faiss"    # Use FAISS for GPU acceleration
    chunk_size: 256     # Smaller chunks = faster retrieval
    top_k: 3            # Fewer results = faster processing
```

### 6.2 CPU Optimization

```yaml
# config/cpu.yaml
project:
  device: "cpu"

llm:
  model_name: ""  # Use fallback LLM

knowledge_sources:
  vector_store:
    embedding_model: "BAAI/bge-small-en-v1.5"  # Smaller model
    chunk_size: 256
```

### 6.3 Caching

```yaml
# Enable query caching
layers:
  query_cache:
    enabled: true
    max_size: 1000
    ttl_seconds: 3600
    similarity_threshold: 0.95
```

---

## 7. Troubleshooting

### Common Issues

| Issue | Solution |
|---|---|
| CUDA out of memory | Enable 4-bit quantization, reduce batch size |
| Slow inference | Use GPU, enable caching, reduce top_k |
| High latency | Profile with log analyzer, optimize bottleneck layer |
| Poor accuracy | Train models on domain data, increase top_k |
| Import errors | Check Python version, reinstall dependencies |

### Logs

```bash
# View real-time logs
tail -f results/logs/aarag.log

# Analyze performance
python src/utils/log_analyzer.py --log results/logs/aarag.log --report

# View structured logs
cat logs/queries/*.json | jq .
```
