# Nanotron Training Framework - Understanding Guide

## Overview
Nanotron is Hugging Face's library for pretraining transformer models. It's designed for simplicity, performance, and scalability with support for 3D parallelism (Data Parallel + Tensor Parallel + Pipeline Parallel).

## Core Architecture

### 1. **Main Components**

#### **Training Script (`run_train.py`)**
- Entry point for all training runs
- Loads configuration (YAML or Python)
- Initializes `DistributedTrainer`
- Sets up dataloaders
- Executes training loop

#### **Trainer (`src/nanotron/trainer.py`)**
- Core training orchestration
- Handles distributed training setup
- Manages checkpointing
- Implements training loop with gradient accumulation
- Supports 3D parallelism (DP, TP, PP)

#### **Configuration System (`src/nanotron/config/`)**
- YAML-based configuration
- Defines model architecture, training hyperparameters, parallelism strategy
- Supports multiple data stages for curriculum learning

#### **Data Pipeline (`src/nanotron/data/`)**
- Three data loading modes:
  1. **Dummy data** (for testing)
  2. **HuggingFace datasets** (for standard datasets)
  3. **Nanosets** (for large-scale tokenized data)

### 2. **Key Concepts**

#### **3D Parallelism**
```
Total GPUs = DP × TP × PP
```

- **Data Parallelism (DP)**: Replicates model across GPUs, each processes different data
- **Tensor Parallelism (TP)**: Splits individual layers/tensors across GPUs
- **Pipeline Parallelism (PP)**: Splits model layers across GPUs in stages

#### **Batch Size Calculation**
```
Global Batch Size = micro_batch_size × batch_accumulation_per_replica × DP
```

- `micro_batch_size`: Samples per GPU per forward pass
- `batch_accumulation_per_replica`: Gradient accumulation steps
- `DP`: Number of data parallel replicas

#### **Training Stages**
Nanotron supports multi-stage training (curriculum learning):
- Each stage can have different datasets, learning rates, etc.
- Defined in `data_stages` in config
- Useful for progressive training strategies

### 3. **File Structure**

```
Smoltron/
├── run_train.py              # Main training script
├── run_generate.py           # Generation/inference script
├── run_evals.py              # Evaluation script
├── slurm_launcher.py         # SLURM cluster launcher
├── examples/                 # Example configs and scripts
│   ├── config_tiny_llama.yaml
│   ├── config_qwen.yaml
│   ├── custom-dataloader/
│   ├── moe/                  # Mixture of Experts
│   └── mamba/                # Mamba architecture
├── src/nanotron/
│   ├── trainer.py            # Core trainer class
│   ├── config/               # Configuration classes
│   ├── data/                 # Data loading utilities
│   ├── models/               # Model implementations
│   ├── parallel/             # Parallelism implementations
│   ├── optim/                # Optimizers
│   └── serialize/            # Checkpointing
└── docs/                     # Documentation
```

### 4. **Training Flow**

```
1. Parse config file (YAML/Python)
   ↓
2. Initialize DistributedTrainer
   - Set up distributed process groups
   - Initialize model with parallelism
   - Create optimizer
   - Load checkpoint (if resuming)
   ↓
3. Create dataloader(s)
   - Based on data_stages config
   - Handle consumed samples for resumption
   ↓
4. Training loop
   - Forward pass
   - Backward pass
   - Gradient accumulation
   - Optimizer step
   - Logging & checkpointing
   ↓
5. Save final checkpoint
```

### 5. **Configuration Deep Dive**

#### **Model Configuration**
```yaml
model:
  dtype: bfloat16              # Training precision
  model_config:
    hidden_size: 2048          # Model dimension
    num_hidden_layers: 28      # Number of transformer layers
    num_attention_heads: 16    # Attention heads
    num_key_value_heads: 2     # GQA (Grouped Query Attention)
    intermediate_size: 11008   # FFN dimension
    vocab_size: 128256         # Vocabulary size
```

#### **Parallelism Configuration**
```yaml
parallelism:
  dp: 4                        # Data parallel size
  tp: 2                        # Tensor parallel size
  pp: 1                        # Pipeline parallel size
  tp_mode: REDUCE_SCATTER      # TP communication mode
  pp_engine: 1f1b              # Pipeline schedule (1F1B or AFAB)
```

#### **Optimizer Configuration**
```yaml
optimizer:
  learning_rate_scheduler:
    learning_rate: 0.0003
    lr_warmup_steps: 2
    lr_decay_style: cosine
  optimizer_factory:
    name: adamW
    adam_beta1: 0.9
    adam_beta2: 0.95
  zero_stage: 0                # ZeRO optimization stage
```

#### **Data Configuration**
```yaml
data_stages:
- data:
    dataset: null              # null = dummy data
    num_loading_workers: 1
    seed: 42
  name: Stable Training Stage
  start_training_step: 1
```

### 6. **Key Features**

✅ **Implemented:**
- 3D parallelism (DP+TP+PP)
- Expert parallelism for MoEs
- AFAB and 1F1B pipeline schedules
- ZeRO-1 optimizer
- FP32 gradient accumulation
- Custom checkpointing
- Spectral µTransfer
- Flash Attention support

🚧 **Roadmap:**
- FP8 training
- ZeRO-3 (FSDP)
- torch.compile support
- Ring attention

### 7. **Supported Models**

- **LLaMA** family (including tiny variants)
- **Qwen2** 
- **Mamba** (state-space models)
- **MoE** (Mixture of Experts)

### 8. **Important Environment Variables**

```bash
CUDA_DEVICE_MAX_CONNECTIONS=1  # Required for distributed ops
WANDB_MODE=disabled            # Disable W&B logging
NANOTRON_BENCHMARK=1           # Enable benchmarking mode
```

### 9. **Checkpointing System**

- Saves model, optimizer, and training state
- Supports resumption from any checkpoint
- Tracks consumed samples per data stage
- Can upload to S3 automatically
- Format: `checkpoints/{step}/`

### 10. **Data Loading Modes**

#### **Mode 1: Dummy Data (Testing)**
```yaml
data:
  dataset: null
```
- Generates random tensors
- Perfect for pipeline testing
- No actual data needed

#### **Mode 2: HuggingFace Datasets**
```yaml
data:
  dataset:
    hf_dataset_or_datasets: "HuggingFaceFW/fineweb"
    text_column_name: "text"
```
- Uses `datasets` library
- Automatic tokenization
- Supports pretraining and SFT

#### **Mode 3: Nanosets (Production)**
```yaml
data:
  dataset:
    dataset_folder: ["path/to/tokenized/data"]
    dataset_weights: [1.0]
```
- Pre-tokenized binary data
- Fastest loading
- Best for large-scale training

## Testing Strategy for Single RTX 3090

### Hardware Constraints
- **GPU**: RTX 3090 (24GB VRAM)
- **Goal**: Test training pipeline, not full-scale training
- **Approach**: Use minimal model with dummy data

### Recommended Test Configuration
1. **Tiny model** (fits in 24GB)
2. **No parallelism** (DP=1, TP=1, PP=1)
3. **Dummy data** (fastest, no preprocessing)
4. **Short training** (100-200 steps)
5. **Frequent checkpointing** (verify save/load)

## Next Steps

1. ✅ Understand the codebase structure
2. 🔄 Create test configuration for RTX 3090
3. 🔄 Run initial test
4. 🔄 Verify checkpointing works
5. 🔄 Test with real data
6. 🔄 Experiment with model sizes

## Useful Commands

```bash
# Basic training
CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun --nproc_per_node=1 run_train.py --config-file config.yaml

# With debugging
CUDA_DEVICE_MAX_CONNECTIONS=1 WANDB_MODE=disabled torchrun --nproc_per_node=1 run_train.py --config-file config.yaml

# Generation from checkpoint
torchrun --nproc_per_node=1 run_generate.py --ckpt-path checkpoints/100/ --tp 1 --pp 1

# Check GPU usage
nvidia-smi -l 1
```

## Common Issues & Solutions

1. **OOM (Out of Memory)**
   - Reduce `micro_batch_size`
   - Reduce `hidden_size` or `num_hidden_layers`
   - Enable gradient checkpointing

2. **Slow data loading**
   - Use Nanosets for large datasets
   - Increase `num_loading_workers`
   - Use dummy data for testing

3. **Checkpoint issues**
   - Ensure `checkpoints_path` directory exists
   - Check disk space
   - Verify parallelism matches on resume

## Resources

- [Official Docs](https://github.com/huggingface/nanotron)
- [Ultrascale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook)
- [Examples Directory](./examples/)
