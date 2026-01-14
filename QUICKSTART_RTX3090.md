# Quick Start Guide for Testing Nanotron on RTX 3090

## Prerequisites Check

Before running, ensure you have:

1. **Python Environment** (3.10 or 3.11)
2. **PyTorch with CUDA** installed
3. **Nanotron dependencies** installed
4. **RTX 3090** with CUDA drivers

## Installation Steps

### 1. Create Virtual Environment (if not done)

```bash
cd /Users/visheshtripathi/Documents/smolLMs/Smoltron

# Using uv (recommended)
uv venv nanotron --python 3.11 && source nanotron/bin/activate && uv pip install --upgrade pip

# OR using venv
python3 -m venv nanotron
source nanotron/bin/activate
pip install --upgrade pip
```

### 2. Install PyTorch with CUDA

```bash
# For CUDA 12.4
uv pip install torch --index-url https://download.pytorch.org/whl/cu124

# OR for CUDA 11.8
uv pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### 3. Install Nanotron

```bash
# Core dependencies
uv pip install -e .

# Additional dependencies for examples
uv pip install datasets transformers datatrove[io] numba wandb

# Optional: Flash Attention (recommended for speed)
# This may take a while to compile
uv pip install ninja triton "flash-attn>=2.5.0" --no-build-isolation
```

## Quick Test Run

### Option 1: Using the Test Script (Easiest)

```bash
# Make script executable
chmod +x test_rtx3090.sh

# Run the test
./test_rtx3090.sh
```

### Option 2: Manual Run

```bash
# Set environment variables
export CUDA_DEVICE_MAX_CONNECTIONS=1
export WANDB_MODE=disabled

# Run training
torchrun --nproc_per_node=1 run_train.py --config-file examples/config_test_rtx3090.yaml
```

## What to Expect

### Training Output

You should see output like:

```
[INFO] Starting training...
[INFO] Iteration 1/100 | Loss: 10.234 | LR: 0.00003
[INFO] Iteration 2/100 | Loss: 10.123 | LR: 0.00006
...
[INFO] Saving checkpoint at step 50...
[INFO] Iteration 50/100 | Loss: 8.456 | LR: 0.0003
...
[INFO] Training completed!
```

### Memory Usage

Expected GPU memory usage:
- **Model**: ~2-3 GB
- **Optimizer states**: ~2-3 GB
- **Activations**: ~1-2 GB
- **Total**: ~6-8 GB (well within 24GB limit)

Monitor with:
```bash
# In another terminal
watch -n 1 nvidia-smi
```

### Training Speed

On RTX 3090, expect:
- **~1-2 seconds per iteration** with the test config
- **Total test time**: ~3-5 minutes for 100 steps

### Checkpoints

Checkpoints will be saved in `checkpoints_test/`:
```
checkpoints_test/
├── 50/              # Checkpoint at step 50
│   ├── model/
│   ├── optimizer/
│   └── metadata.json
└── 100/             # Final checkpoint
    ├── model/
    ├── optimizer/
    └── metadata.json
```

## Verification Steps

### 1. Check Training Completed

```bash
# Should show checkpoint directories
ls -lh checkpoints_test/
```

### 2. Verify Checkpoint Contents

```bash
# Check step 50 checkpoint
ls -lh checkpoints_test/50/
```

### 3. Test Generation (Optional)

```bash
torchrun --nproc_per_node=1 run_generate.py \
  --ckpt-path checkpoints_test/100/ \
  --tp 1 \
  --pp 1
```

## Understanding the Test Configuration

The `config_test_rtx3090.yaml` is optimized for testing:

| Parameter | Value | Reason |
|-----------|-------|--------|
| `hidden_size` | 1024 | Smaller model (~500M params) |
| `num_hidden_layers` | 12 | Fewer layers for faster iteration |
| `sequence_length` | 512 | Shorter sequences for speed |
| `micro_batch_size` | 2 | Small batch for testing |
| `train_steps` | 100 | Quick test run |
| `dataset` | null | Dummy data (no preprocessing) |
| `dp/tp/pp` | 1/1/1 | Single GPU, no parallelism |

**Estimated Model Size**: ~500M parameters (~2GB in bfloat16)

## Next Steps After Successful Test

### 1. Test with Real Data

Modify the config to use HuggingFace datasets:

```yaml
data_stages:
- data:
    dataset:
      hf_dataset_or_datasets: "HuggingFaceFW/fineweb-edu"
      hf_dataset_splits: "train"
      text_column_name: "text"
      dataset_processing_num_proc_per_process: 4
    num_loading_workers: 2
    seed: 42
  name: Real Data Training
  start_training_step: 1
```

### 2. Scale Up Model Size

Gradually increase model size while monitoring memory:

```yaml
model_config:
  hidden_size: 2048        # Increase from 1024
  num_hidden_layers: 24    # Increase from 12
  num_attention_heads: 16  # Increase from 8
```

### 3. Optimize Batch Size

Find maximum batch size that fits in memory:

```yaml
tokens:
  micro_batch_size: 4      # Try 4, 8, 16...
  sequence_length: 1024    # Try 1024, 2048...
```

### 4. Resume Training Test

Test checkpoint resumption:

```bash
# Stop training mid-way (Ctrl+C)
# Then resume with:
torchrun --nproc_per_node=1 run_train.py \
  --config-file examples/config_test_rtx3090.yaml
```

Update config:
```yaml
checkpoints:
  resume_checkpoint_path: checkpoints_test/50/
```

## Troubleshooting

### Issue: OOM (Out of Memory)

**Solution**: Reduce batch size or model size
```yaml
tokens:
  micro_batch_size: 1      # Reduce from 2
  sequence_length: 256     # Reduce from 512
```

### Issue: "CUDA out of memory" during backward pass

**Solution**: Enable gradient checkpointing (saves memory at cost of speed)
```yaml
model:
  use_gradient_checkpointing: true
```

### Issue: Slow training

**Solution**: 
1. Install Flash Attention (if not already)
2. Increase batch size
3. Use longer sequences (more efficient)

### Issue: Import errors

**Solution**: Ensure all dependencies installed
```bash
uv pip install -e .
uv pip install datasets transformers
```

### Issue: "No module named 'nanotron'"

**Solution**: Install in editable mode
```bash
cd /Users/visheshtripathi/Documents/smolLMs/Smoltron
uv pip install -e .
```

## Performance Optimization Tips

### 1. Use Flash Attention
```bash
uv pip install flash-attn --no-build-isolation
```

### 2. Use Fused Kernels
```yaml
optimizer:
  optimizer_factory:
    torch_adam_is_fused: true  # Already enabled
```

### 3. Optimize Data Loading
```yaml
data:
  num_loading_workers: 4  # Increase for real data
```

### 4. Use bfloat16
```yaml
model:
  dtype: bfloat16  # Already enabled, faster than fp32
```

## Monitoring Training

### 1. TensorBoard (Optional)

```bash
# Enable profiler in config
profiler:
  profiler_export_path: ./tb_logs/

# View in TensorBoard
tensorboard --logdir tb_logs/
```

### 2. Weights & Biases (Optional)

```bash
# Login
wandb login

# Enable in environment
export WANDB_MODE=online

# Set project in config
general:
  project: my_nanotron_project
```

### 3. GPU Monitoring

```bash
# Real-time monitoring
watch -n 1 nvidia-smi

# Log to file
nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv -l 1 > gpu_usage.log
```

## Expected Timeline

1. **Installation**: 10-15 minutes (first time)
2. **First test run**: 3-5 minutes
3. **Verification**: 1-2 minutes
4. **Total**: ~20 minutes to get up and running

## Success Criteria

✅ Training starts without errors  
✅ GPU memory usage is stable  
✅ Loss decreases over iterations  
✅ Checkpoints are saved correctly  
✅ Training completes all 100 steps  

## Additional Resources

- **Main Documentation**: [README.md](README.md)
- **Understanding Guide**: [NANOTRON_UNDERSTANDING.md](NANOTRON_UNDERSTANDING.md)
- **Your First Training**: [docs/your-first-training.md](docs/your-first-training.md)
- **Examples**: [examples/](examples/)

## Questions?

Common questions answered in [NANOTRON_UNDERSTANDING.md](NANOTRON_UNDERSTANDING.md):
- How does 3D parallelism work?
- What's the difference between data loading modes?
- How to calculate memory requirements?
- How to scale to multiple GPUs?
