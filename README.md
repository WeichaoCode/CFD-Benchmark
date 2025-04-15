# Code Generation Benchmark Framework
This repository provides a benchmark suite to evaluate the performance of large language models (LLMs) on
**code generation tasks for solving partial differential equations (PDEs)**. It includes prompt templates, 
auto-evaluation utilities, and solution validation for physics-based problems.

---

## 📁 Repository Structure

```
├── __pycache__
│   ├── utils.cpython-310.pyc
│   └── utils.cpython-38.pyc
├── compare     # compare the MSE
├── compare_images # compare the fluid flow images
│   ├── ground_truth   
│   ├── prediction
│   └── table
├── image  # save the fluid flow images
│   ├── gemini
│   ├── gpt-4o
│   ├── haiku
│   ├── o3-mini
│   └── sonnet-35
├── prompt
│   ├── PDE_TASK_QUESTION_ONLY.json
│   └── prompts.json
├── report # log files and report
├── results # .npy files
│   ├── prediction
│   └── solution # groud truth 
├── solution # ground truth python code
├── solver # generate python code 
│   ├── gemini
│   ├── gpt-4o
│   ├── haiku
│   ├── o3-mini
│   └── sonnet-35
└── table # create table contains MSE ...
```
---

## 🚀 Quickstart

### 1. 🛠️ Installation

We recommend using **conda** to manage dependencies:

```bash
conda create -n pde_benchmark python=3.10
conda activate pde_benchmark

# Install Python dependencies
pip install -r requirements.txt
```
Make sure to install extra dependencies manually if needed:
```bash
pip install openai boto3 matplotlib scikit-learn opencv-python scikit-image
```

### 2. 🔑 API Keys
Before running, make sure your ```OpenAI```, ```Gemini```, or ```Bedrock``` API credentials are set.

You can either export them in environment variables:
```bash
export OPENAI_API_KEY=your-key
export GEMINI_API_KEY=your-key
```
Or directly edit the ```utils.py``` to insert your keys.

## 💻 Run the Benchmark
To run the benchmark pipeline for all prompts and models:
```bash
python train_plain_prompts.py
```
