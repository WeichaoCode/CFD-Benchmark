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
This script performs the following steps:

* **Load Prompts**  
  Loads each task prompt from the `prompt/` directory.

* **Call LLM to Generate Code**  
  Uses the selected Large Language Model (LLM) (e.g., GPT-4o, Gemini, Claude, etc.) to generate solver code for the given PDE problem.

* **Save Generated Code**  
  Saves the generated Python solver scripts into the `solver/` directory.

* **Execute and Evaluate**  
  Runs each solver, compares the results with the ground-truth reference in the `solution/` directory.

* **Log Results**  
  * Performance metrics and runtime are logged into the `report/` folder.  
  * Visual outputs are saved in the `image/` folder.  
  * Final comparison results are stored in the `results/` directory.
  * Final comparison results tables are stored in the `table/` directory.

## 📊 Output and Evaluation
* Execution results (error logs, pass/fail): saved in `report/`

* Generated solvers: `solver/`

* Reference solutions: `solution/`

* Performance images: `image/`

* Comparison tools: `compare/`, `compare_images/`

* Summary tables: `table/`

## 🧪 Add Your Own Task
* Add a new `prompt_name.json` under `prompt/`

* Add the corresponding ground truth solution under `solution/`
