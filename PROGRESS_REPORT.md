# Nanotron Training Test - Progress Report

## ✅ What We've Accomplished

### 1. **Installation Complete**
- ✅ Python 3.11.14 environment
- ✅ PyTorch 2.3.1 + CUDA 12.1
- ✅ Flash Attention 2.6.3 (compiled and working!)
- ✅ All Nanotron dependencies installed

### 2. **Code Fixes Applied**
- ✅ Fixed `StandardParametrizator` to accept `ModelArgs` instead of `Config`
  - File: `src/nanotron/scaling/parametrization.py`
  - Issue: Type mismatch in initialization
  - Solution: Changed parameter type and attribute access paths

### 3. **Training Pipeline Progress**
The training got much further than initial attempts:

```
✅ Model building: SUCCESS (215M parameters)
✅ Parameter initialization: SUCCESS  
✅ Optimizer creation: SUCCESS
✅ Data loading: SUCCESS (dummy data)
✅ Training start: SUCCESS
✅ Forward pass: SUCCESS
❌ Backward pass: FAILED (gradient shape mismatch)
```

## ❌ Current Issue

### Error Details
```
RuntimeError: Function _ColumnLinearNoAsyncCommunicationReduceScatterModeBackward 
returned an invalid gradient at index 0 - got [1024, 1024] but expected shape 
compatible with [512, 2, 1024]
```

### Root Cause Analysis
This appears to be a **tensor shape mismatch in the backward pass** of tensor parallel linear layers. Possible causes:

1. **GQA (Grouped Query Attention) Bug**: Using `num_key_value_heads=2` with `num_attention_heads=8` might trigger edge cases
2. **Sequence Length Issue**: The gradient reshaping might not handle all sequence lengths correctly
3. **Version Compatibility**: This specific Nanotron commit might have a bug that's fixed in newer versions

## 🔧 Troubleshooting Steps Taken

### Attempt 1: Original RTX 3090 Config
- Model: 215M params (1024 hidden, 12 layers)
- Sequence: 512
- GQA: 8 heads, 2 KV heads
- **Result**: Gradient shape error

### Attempt 2: Simplified Config (Created)
- Model: ~20M params (256 hidden, 4 layers)  
- Sequence: 128 (shorter)
- No GQA: 4 heads, 4 KV heads (1:1 ratio)
- **Status**: Ready to test

## 🎯 Next Steps

### Option 1: Test Simplified Config (Recommended)
```bash
cd /workspace/Smoltron
CUDA_DEVICE_MAX_CONNECTIONS=1 WANDB_MODE=disabled \
  torchrun --nproc_per_node=1 run_train.py \
  --config-file examples/config_test_simple.yaml
```

**Why this might work:**
- No GQA (avoids potential GQA bugs)
- Shorter sequence (128 vs 512)
- Smaller model (faster to debug)
- Simpler architecture

### Option 2: Update Nanotron
```bash
cd /workspace/Smoltron
git pull origin main
# Or checkout a specific stable tag
git checkout v0.4  # if available
```

### Option 3: Use Real Data Instead of Dummy
The dummy data generator might have bugs. Try with actual HuggingFace dataset:

```yaml
data_stages:
- data:
    dataset:
      hf_dataset_or_datasets: "roneneldan/TinyStories"
      hf_dataset_splits: "train"
      text_column_name: "text"
      dataset_processing_num_proc_per_process: 2
    num_loading_workers: 2
    seed: 42
  name: Real Data Test
  start_training_step: 1
```

### Option 4: Check for Known Issues
```bash
# Search for similar issues in Nanotron repo
cd /workspace/Smoltron
git log --all --grep="gradient" --grep="backward" --oneline | head -20
```

## 📊 System Information

### Hardware
- **GPU**: NVIDIA GeForce RTX 3090 (24GB)
- **CPU**: AMD Ryzen 5 3600 (6-core, 12-thread)
- **RAM**: Sufficient for training

### Software Versions
```
PyTorch: 2.3.1+cu121
Flash-Attn: 2.6.3
Transformers: 4.57.5
Nanotron: 0.4
CUDA: 12.1 (PyTorch) / 12.8 (Runtime)
```

### Memory Usage (at failure point)
- Model: ~415 MB
- Peak allocated: ~2.06 GB
- Peak reserved: ~2.13 GB
- **Plenty of headroom on 24GB GPU!**

## 🔍 What We Learned About Nanotron

### Architecture Understanding
1. **3D Parallelism**: DP × TP × PP (we're using 1×1×1)
2. **Pipeline**: Model building → Init → Optimizer → Data → Training loop
3. **Dummy Data**: Generates random tensors for testing
4. **Checkpointing**: Saves every N steps

### Key Files
- `run_train.py`: Main entry point
- `src/nanotron/trainer.py`: Core training logic
- `src/nanotron/scaling/parametrization.py`: Weight initialization
- `src/nanotron/data/dataloader.py`: Data loading
- `src/nanotron/parallel/`: Parallelism implementations

### Configuration System
- YAML-based with strong typing
- Supports multiple training stages
- Flexible parallelism configuration
- Easy to customize

## 💡 Recommendations

### For Immediate Testing
1. **Try the simplified config first** (`config_test_simple.yaml`)
   - If it works: Gradually increase complexity
   - If it fails: The issue is deeper than config

2. **Check Nanotron version**
   - Current: v0.4
   - Latest might have fixes

3. **Try without GQA**
   - Set `num_key_value_heads = num_attention_heads`
   - Eliminates one variable

### For Understanding the Codebase
1. ✅ Read `NANOTRON_UNDERSTANDING.md` (created earlier)
2. ✅ Study the config files in `examples/`
3. ✅ Look at `docs/your-first-training.md`
4. 🔄 Trace through `trainer.py` to understand training loop
5. 🔄 Understand the parallelism implementations

### For Production Use
1. Start with proven configs (like `config_tiny_llama.yaml`)
2. Scale up gradually
3. Use real datasets (not dummy data)
4. Enable checkpointing and logging
5. Monitor GPU memory and utilization

## 📝 Files Created

1. `NANOTRON_UNDERSTANDING.md` - Comprehensive framework guide
2. `QUICKSTART_RTX3090.md` - Installation and testing guide
3. `examples/config_test_rtx3090.yaml` - Original test config (has issues)
4. `examples/config_test_simple.yaml` - Simplified test config (try this!)
5. `test_rtx3090.sh` - Automated test script

## 🐛 Known Issues

1. **Gradient Shape Mismatch**: Current blocker
   - Affects: Backward pass in tensor parallel layers
   - Workaround: Try simpler config or update Nanotron

2. **Flash-Attn Compatibility**: Resolved
   - Required: PyTorch 2.3.1 + Flash-Attn 2.6.3
   - Don't use: PyTorch 2.6.0 (too new, incompatible)

3. **NumPy Version**: Watch out
   - Nanotron wants: numpy<2
   - Datatrove installs: numpy 2.x
   - Solution: `uv pip install "numpy<2"`

## ✨ Success Criteria

To consider the test successful, we need:
- [ ] Training completes at least 10 steps
- [ ] Loss decreases over iterations
- [ ] Checkpoints save correctly
- [ ] No OOM errors
- [ ] Gradients flow correctly (backward pass works)

## 🚀 Next Command to Run

```bash
# Test with simplified config
cd /workspace/Smoltron
CUDA_DEVICE_MAX_CONNECTIONS=1 WANDB_MODE=disabled \
  torchrun --nproc_per_node=1 run_train.py \
  --config-file examples/config_test_simple.yaml
```

If this works, you'll see:
```
[INFO]: Iteration 1/20 | Loss: X.XXX
[INFO]: Iteration 2/20 | Loss: X.XXX (should decrease)
...
[INFO]: Saving checkpoint at step 10
```

Good luck! 🎉
