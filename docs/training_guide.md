# AARAG Training Guide

## 1. Overview

AARAG has three trainable components:
1. **Query Complexity Classifier** (Layer 2) — DeBERTa-v3-base
2. **Retrieval Evaluator** (Layer 3) — DeBERTa-v3-small
3. **Reflection Token Generator** (Layer 4) — Llama-3.1-8B with LoRA

---

## 2. Data Preparation

### 2.1 Running Data Preparation

```bash
python src/training/data_preparation.py
```

### 2.2 Data Sources

**Router Training Data:**
- Level 0 (No Retrieval): TriviaQA simple questions
- Level 1 (Single-Step): Natural Questions
- Level 2 (Multi-Step): HotpotQA multi-hop questions
- Level 3 (Graph-Global): Synthetic corpus-level questions

**Evaluator Training Data:**
- Positive pairs: (query, relevant_document) from HotpotQA
- Negative pairs: (query, random_document) from shuffled corpus

**Reflection Training Data:**
- Synthetic: 2000 samples with reflection token annotations
- Scenarios: relevant_supported, relevant_partial, irrelevant, no_context

### 2.3 Custom Data

To use your own data, create JSON files in this format:

```json
[
  {"query": "Your question here", "label": 0, "source": "custom"},
  {"query": "Another question", "label": 1, "source": "custom"}
]
```

---

## 3. Training the Router

### 3.1 Command

```bash
python src/training/train_router.py \
    --train-data data/processed/router_train.json \
    --val-data data/processed/router_val.json \
    --model-name microsoft/deberta-v3-base \
    --output-dir models/router \
    --epochs 10 \
    --batch-size 32 \
    --lr 2e-5
```

### 3.2 Hyperparameters

| Parameter | Default | Description |
|---|---|---|
| epochs | 10 | Training epochs |
| batch_size | 32 | Batch size |
| lr | 2e-5 | Learning rate |
| warmup_ratio | 0.1 | Warmup steps ratio |
| weight_decay | 0.01 | L2 regularization |

### 3.3 Expected Output

```
Epoch 1/10 — Loss: 1.2340, Val Accuracy: 0.6500
  Saved best model (accuracy: 0.6500)
Epoch 2/10 — Loss: 0.8920, Val Accuracy: 0.7200
  Saved best model (accuracy: 0.7200)
...
Epoch 10/10 — Loss: 0.2340, Val Accuracy: 0.8700
  Saved best model (accuracy: 0.8700)

Training complete. Best accuracy: 0.8700
Model saved to: models/router
```

### 3.4 Monitoring

```bash
# View training logs
tensorboard --logdir models/router/runs
```

---

## 4. Training the Retrieval Evaluator

### 4.1 Command

```bash
python src/training/train_evaluator.py \
    --train-data data/processed/evaluator_train.json \
    --val-data data/processed/evaluator_val.json \
    --model-name microsoft/deberta-v3-small \
    --output-dir models/evaluator \
    --epochs 5 \
    --batch-size 16 \
    --lr 3e-5
```

### 4.2 Expected Output

```
Epoch 1/5 — Loss: 0.6890, Val F1: 0.7200
  Saved best model (F1: 0.7200)
...
Epoch 5/5 — Loss: 0.2340, Val F1: 0.9100
  Saved best model (F1: 0.9100)

Training complete. Best F1: 0.9100
```

---

## 5. Training Reflection Tokens (LoRA)

### 5.1 Requirements

- GPU with 16GB+ VRAM (RTX 4090, A100)
- Llama-3.1-8B-Instruct model access (HuggingFace)

### 5.2 Command

```bash
python src/training/train_reflection.py \
    --train-data data/processed/reflection_train.json \
    --model-name meta-llama/Llama-3.1-8B-Instruct \
    --output-dir models/reflection \
    --epochs 3 \
    --batch-size 4 \
    --lr 1e-5 \
    --lora-rank 16
```

### 5.3 LoRA Configuration

```yaml
lora_config:
  r: 16                    # LoRA rank
  lora_alpha: 32           # Scaling factor
  target_modules:          # Which layers to adapt
    - q_proj
    - v_proj
    - k_proj
    - o_proj
  lora_dropout: 0.05
  bias: none
  task_type: CAUSAL_LM
```

### 5.4 Expected Output

```
Trainable params: 4,194,304 || All params: 8,030,261,248 || Trainable%: 0.0522
Epoch 1/3 — Loss: 1.5670
Epoch 2/3 — Loss: 0.8920
Epoch 3/3 — Loss: 0.5670

Training complete. Model saved to: models/reflection
```

---

## 6. Full Training Pipeline

### 6.1 Automated Script

```bash
#!/bin/bash
# Full AARAG training pipeline

echo "=== Phase 1: Data Preparation ==="
python src/training/data_preparation.py

echo "=== Phase 2: Train Router ==="
python src/training/train_router.py --epochs 10

echo "=== Phase 3: Train Evaluator ==="
python src/training/train_evaluator.py --epochs 5

echo "=== Phase 4: Train Reflection ==="
python src/training/train_reflection.py --epochs 3

echo "=== Training Complete ==="
```

### 6.2 Training Time Estimates

| Component | GPU | Time |
|---|---|---|
| Router | RTX 4090 | ~30 minutes |
| Evaluator | RTX 4090 | ~15 minutes |
| Reflection (LoRA) | RTX 4090 | ~2 hours |
| Reflection (LoRA) | A100 | ~45 minutes |
| Full pipeline | RTX 4090 | ~3 hours |

---

## 7. Fine-Tuning Tips

### 7.1 Learning Rate

- Start with 2e-5 for classification tasks
- Use 1e-5 for generation tasks (LoRA)
- Use linear warmup (10% of total steps)

### 7.2 Batch Size

- Larger batches (32-64) for classification
- Smaller batches (4-8) for generation (memory constraints)
- Use gradient accumulation if needed

### 7.3 Overfitting Prevention

- Monitor validation loss
- Use early stopping (patience=3)
- Apply dropout (0.05-0.1)
- Use weight decay (0.01)

### 7.4 Data Augmentation

- Paraphrase questions for router training
- Swap entities for evaluator training
- Vary context length for reflection training

---

## 8. Using Trained Models

### 8.1 Loading Models

```python
from src.aarag import AARAG

aarag = AARAG(config_path="config/default.yaml")
aarag.load("models/")
```

### 8.2 Individual Components

```python
from src.layers.adaptive_router import AdaptiveRouter

router = AdaptiveRouter()
router.load("models/router")

decision = router.route("What is machine learning?")
print(decision.strategy)  # "single_step"
```

---

## 9. Troubleshooting

| Issue | Solution |
|---|---|
| CUDA OOM | Reduce batch size, enable 4-bit quantization |
| Slow training | Use mixed precision (fp16), increase batch size |
| Poor accuracy | More training data, higher learning rate |
| Overfitting | More regularization, early stopping |
| Model not found | Check model path, run download_models.py |
