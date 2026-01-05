# Quickstart

1) Install dependencies:
```
pip install -r requirements.txt
```

2) Download the dataset into the repo root (so it sits alongside prompts/ and src/):
```
git clone https://huggingface.co/datasets/Shayfra7926/PANOPTICON
```

3) Generate prompts (single chunk example):
```
python src/prompt_gen.py --num-chunks 7 --chunk-index 0
```

4) Run the Prompt Inversion Attack (PIA):
```
python src/inverter.py
```

# Recommended resources
Python 3.10+
200G Memory
2 A100 GPUs

# Overview
This repository is organized into three required top-level directories:

- PANOPTICON_v1.4/ — dataset (downloaded separately)
- prompts/ — system prompt templates
- src/ — code
  - utils/ — dataset helpers + evaluation utilities

Recommended layout:

```text
repo-root/
├─ PANOPTICON_v1.4/
├─ prompts/
└─ src/
   ├─ prompt_gen.py
   ├─ inverter.py
   └─ utils/
```
## Panopticon_v1.4
To utilize the PANOPTICON dataset you must first download it from HuggingFace

```
git clone https://huggingface.co/datasets/Shayfra7926/PANOPTICON
```
It should exist at the top level with prompts and src.

## Prompts
The "prompts" folder contains the system prompts needed to repeat the output of the PANOPTICON dataset. Primarily, gen_prompts.md provides the template for generating PII laden prompts while benign_prompt is a variant that handles benign (non-PII laden) prompt outputs. "response.md" is an experimental prompt for generating associated responses based on the prompt. This can enable prompt-response pairs. 

## src
This folder contains the primary code. prompt_gen.py will run the actual prompt generation. 
It accepts three arguments:
1. Number of Chunks
2. Chunk index
3. Model name

1 and 2 help run batch jobs, where number of chunks determines number of splits while index ensures each split contains the appropriate index range. Model name, defaulted to meta-llama/Llama-3.1-8B-Instruct, enables model selection. 

inverter.py will run the Prompt Inversion Attack (PIA) and output relevant statistics. 

### utils
src contains the subfolder "utils," they handle the PANORAMA and PANORAMA+ datasets. Further, it contains code for running evaluation metrics. 

# Environment
```
pip install -r requirements.txt 
```

# Example Script Usage
```
#!/bin/bash
#SBATCH --job-name=panopticon
#SBATCH --gpus=2
#SBATCH --mem=200G
#SBATCH --ntasks-per-node=1
#SBATCH --time=20:00:00
#SBATCH --array=0-6   # 7 jobs: 0,1,2,3,4,5,6

# Activate your environment (example; replace with your own)
source /path/to/venv/bin/activate

# Move to repository root (replace with your own)
cd /path/to/PANOPTICON-DataGen/

# SLURM_ARRAY_TASK_ID will be 0..6
python src/prompt_gen.py \
  --num-chunks 7 \
  --chunk-index ${SLURM_ARRAY_TASK_ID}
```
