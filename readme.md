# PVSDL: A Vision-Language Model Evaluation Framework for Solar Panel Dust Detection

[![Dataset](https://img.shields.io/badge/Dataset-HuggingFace-blue)](https://huggingface.co/datasets/asdmd/PVSDL_DATASET)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-green)](https://www.python.org/)

## Overview

The **PV Soiling Detection with LLMs (PVSDL) framework** is an automated benchmarking framework designed to evaluate the performance of state-of-the-art **Vision-Language Models (VLMs)** in identifying soiling and dust accumulation on solar panels. 

Efficient Operation and Maintenance (O&M) of photovoltaic systems is critical for energy yield. This tool allows researchers to systematically test various VLMs (currently supported: `gemini-3-flash`, `gemini-3.1-pro`, `kimi-k2.5`, `grok-4.1-fast`, `gpt-4o-mini`, and `gemini-2.5-flash`) using different prompt engineering strategies to determine their diagnostic accuracy and reliability in real-world scenarios.

## Key Features

* **Multi-Model Benchmarking**: Integrated with the OpenRouter API to support seamless switching between various proprietary and open-source VLMs.
* **Prompt Engineering Suite**: Supports comparative analysis of different prompt templates (`basic`, `detailed`, `cot`, and `technical`) to evaluate instruction-following capabilities.
* **Statistical Robustness**: Includes a **Repeated Test Mode** specifically designed for scientific validation, generating variance and distribution data (ideal for Violin Plots or Box Plots).
* **Automated Metrics**: Built-in calculation of binary classification metrics: True Positive (TP), True Negative (TN), False Positive (FP), and False Negative (FN).
* **Data Integrity**: Real-time CSV logging and raw JSON response storage for full reproducibility of experimental results.

---

## Installation & Setup

### 1. Repository & Dataset
Clone the main repository and the corresponding dataset from HuggingFace:

```bash
# Clone the framework
git clone [https://github.com/hengyue-dunhuang/PVSDL.git](https://github.com/hengyue-dunhuang/PVSDL.git)
cd PVSDL

# Clone the dataset into the project folder
git clone [https://huggingface.co/datasets/asdmd/PVSDL_DATASET](https://huggingface.co/datasets/asdmd/PVSDL_DATASET)
```

### 2. Environment Configuration
We recommend using **Python 3.12.12**. You can set up the environment using Conda or pip:

**Option A: Conda (Recommended)**
```bash
conda env create -f environment.yml
conda activate PVSDL
```

**Option B: pip**
```bash
pip install -r requirements.txt
```

### 3. API Configuration
1. Register an account at [OpenRouter.ai](https://openrouter.ai/).
2. Generate an API Key.
3. Locate the .env file in the root directory of the PVSDL codebase.
4. Open the .env file and directly copy your generated key into it, replacing the placeholder text "input your openrouter API KEY here".

---

## Usage Guide

### 1. Exploratory Commands
Before running experiments, verify the available configurations:
```bash
# List all supported VLM model IDs
python main.py --list-models

# List available prompt templates
python main.py --list-prompts
```

### 2. Standard Benchmarking (`--test`)
Evaluate multiple models across multiple prompt styles. This is useful for large-scale comparative studies.
```bash
python main.py --test \
    --images-dir PVSDL_DATASET/test \
    --models kimi-k2.5,gpt-4o-mini \
    --prompts basic,detailed \
    --all-images
```

### 3. Statistical Repeatability Test (`--repeat-test`)
For SCI-grade analysis, run a single model/prompt combination multiple times to account for model hallucination or variance.
```bash
python main.py --repeat-test \
    --single-model kimi-k2.5 \
    --single-prompt detailed \
    --repeat 50 \
    --samples 20
```

### Arguments Summary

| Argument | Description | Default |
| :--- | :--- | :--- |
| `--test` | Execute multi-model/multi-prompt matrix test. | - |
| `--repeat-test`| Execute repeated runs for statistical analysis. | - |
| `--images-dir` | Path to the image dataset directory. | `PVSDL_DATASET/test` |
| `--models` | Comma-separated VLM IDs for benchmarking (e.g., `gemini-3.1-pro`, `grok-4.1-fast`). | From config |
| `--prompts` | Comma-separated Prompt IDs (Supported: `basic`, `detailed`, `cot`, `technical`). | `basic` |
| `--repeat` | Number of iterations for repeated tests. | 50 |
| `--samples` | Number of images to sample per run. | 10 |
| `--all-images` | Use the entire dataset for testing. | False |

---

## Data Output & Analysis

Results are stored in two formats:
1.  **Metric Summary**: `vlm_experiment_results.csv` contains the classification counts and timestamps for every run.
2.  **Raw Evidence**: `results/raw_predictions/` contains the full JSON responses from the VLMs, allowing for qualitative analysis of model reasoning.

## Citation
If you use this framework or dataset in your research, please cite our work:
```