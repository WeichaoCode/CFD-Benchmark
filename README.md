# 🌀 CFD Code Generation Benchmark

## 📌 Overview

This project provides a **benchmarking framework** to evaluate the performance of large language models (LLMs) in generating accurate and efficient **CFD (Computational Fluid Dynamics)** solvers using Python. The focus is on finite difference method (FDM) based PDE solvers for classical CFD problems.

It automates the entire pipeline:
- Prompt generation
- Code generation using LLMs
- Code execution and validation
- Numerical error analysis
- Visualization and logging

---

## 🎯 Objectives

- Evaluate how well LLMs (e.g., GPT-4o, DeepSeek, Claude) can generate correct, efficient, and stable CFD code.
- Use a wide range of CFD problems, covering 1D and 2D linear/nonlinear PDEs.
- Automatically compare generated results to ground truth and compute multiple error metrics.
- Support both single-problem testing and batch evaluation.

---

## 📂 Directory Structure



---

## ⚙️ How It Works

1. **Prompt Generation**  
   Each PDE task has a detailed natural language prompt (with mathematical and numerical method details).

2. **Code Generation with LLM**  
   LLMs like GPT-4o are called via OpenAI API to generate the Python code.

3. **Code Execution**  
   The generated code is executed in a sandboxed environment, and `.npy` files are produced as outputs.

4. **Comparison with Ground Truth**  
   Multiple metrics are computed:  
   - MSE (Mean Squared Error)  
   - MAE (Mean Absolute Error)  
   - RMSE  
   - Cosine Similarity  
   - R² Score

5. **Visualization**  
   For each solution, we plot:
   - Ground truth
   - Prediction
   - Absolute error

---

## 📈 Metrics and Evaluation

Each file is compared to its ground truth using:

| Metric              | Description                         |
|---------------------|-------------------------------------|
| MSE                 | Overall squared error               |
| MAE                 | Average absolute error              |
| RMSE                | Root of MSE                         |
| Cosine Similarity   | Shape similarity                    |
| R² Score            | Regression accuracy (1 is perfect)  |

---

## 🧪 Example Supported CFD Problems

- ✅ 1D Linear Convection
- ✅ 1D Heat Equation
- ✅ 1D Euler Shock Tube
- ✅ 2D Diffusion
- ✅ 2D Convection
- ✅ 2D Burgers' Equation
- ✅ 2D Navier-Stokes Cavity Flow
- ✅ 2D Steady/Unsteady Heat Equation

---

## 🚀 Usage

### 🔹 Step 1: Generate Code Using LLM

```bash
python run_generation.py



