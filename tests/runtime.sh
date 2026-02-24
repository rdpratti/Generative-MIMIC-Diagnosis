bin/bash

# Navigate to your script directory
cd ~/projects/Thesis/src

#init conda
source ~/miniconda3/etc/profile.d/conda.sh

# Activate conda environment (if not already activated)
conda activate rag_thesis

echo "Python: $(which python)"
echo "Environment: $CONDA_DEFAULT_ENV"

# Run Python script with arguments
python Gemma_Diagnosis.py \
    --temperature 0.1 \
    --train_ct 100 \
    --test_ct 15 \
    --train_seq_size 500 \
    --test_seq_size 1000 \
    --example_size 500 \
    --example_ct 3 \
    --seed 999 \
    --few_shot_type N \
    --use_rag \
    --balanced_rag \
