#!/bin/bash
# Run full AARAG evaluation

echo "=========================================="
echo "AARAG Evaluation Pipeline"
echo "=========================================="

# Step 1: Prepare data
echo -e "\n[1/4] Preparing evaluation data..."
python src/training/data_preparation.py

# Step 2: Run evaluation
echo -e "\n[2/4] Running evaluation..."
python src/evaluation/evaluate.py --full --max-samples 500

# Step 3: Run ablation analysis
echo -e "\n[3/4] Running ablation analysis..."
python src/evaluation/ablation.py

# Step 4: Generate plots
echo -e "\n[4/4] Generating plots..."
python src/utils/visualization.py

echo -e "\n=========================================="
echo "Evaluation complete!"
echo "Results saved to: results/"
echo "=========================================="
