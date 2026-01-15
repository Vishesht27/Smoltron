#!/bin/bash
CONFIG=$1
NAME=$2

# 1. Run the Training
torchrun --nproc_per_node=4 run_train.py --config-file $CONFIG

# 2. Get the latest checkpoint (e.g., 1526)
LATEST_CKPT=$(ls -td checkpoints_250m/*/ | head -1)

# 3. Convert to HF Format automatically
SAVE_PATH="./hf_models/$NAME"
export PYTHONPATH=$PYTHONPATH:$(pwd)/examples/llama

torchrun --nproc_per_node=1 examples/llama/convert_nanotron_to_hf.py \
  --checkpoint_path=$LATEST_CKPT \
  --save_path=$SAVE_PATH \
  --tokenizer_name="HuggingFaceTB/SmolLM3-3B"

# 4. Trigger your push_to_hf.py script
python push_to_hf.py --model_path=$SAVE_PATH --repo_id="your-username/$NAME"