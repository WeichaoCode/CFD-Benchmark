import os
import json
import re
from openai import OpenAI

# === OpenAI API Configuration ===
api_key = "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"
client = OpenAI(api_key=api_key)

# === Paths ===
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # PDE_Benchmark root
PROMPTS_FILE = os.path.join(ROOT_DIR, "prompts", "PDE_TASK_PROMPT.json")
OUTPUT_FOLDER = os.path.join(ROOT_DIR, "solver/generated_solvers")
USAGE_FILE = os.path.join(OUTPUT_FOLDER, "reports/token_usage_summary.json")

# Ensure the output directory exists
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# === Pricing (USD per 1,000 tokens) ===
INPUT_TOKEN_COST = 0.03
OUTPUT_TOKEN_COST = 0.06

# === Load PDE Prompts ===
with open(PROMPTS_FILE, "r") as file:
    pde_prompts = json.load(file)

# === Token Usage Tracking ===
total_input_tokens = 0
total_output_tokens = 0

# === Process Each Prompt ===
for task_name, prompt in pde_prompts["prompts"].items():
    print(f"🔹 Generating code for: {task_name}...")

    try:
        # Call OpenAI GPT-4 API
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert in CFD and Python."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=7000,
            temperature=0
        )

        # Extract model response
        model_response = response.choices[0].message.content.strip()

        # Extract Python code using regex
        code_match = re.findall(r"```python(.*?)```", model_response, re.DOTALL)
        extracted_code = code_match[0].strip() if code_match else "# No valid Python code extracted"

        # Track token usage
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens

        total_input_tokens += input_tokens
        total_output_tokens += output_tokens

    except Exception as e:
        print(f"❌ Error for {task_name}: {str(e)}")
        extracted_code = f"# Error: {str(e)}"
        input_tokens = 0
        output_tokens = 0

    # === Save Generated Code ===
    filename = f"{task_name}.py".replace(" ", "_")
    file_path = os.path.join(OUTPUT_FOLDER, filename)

    with open(file_path, "w") as code_file:
        code_file.write(extracted_code)

    print(f"✅ Code saved: {file_path}")
    print(f"📊 Tokens - Input: {input_tokens}, Output: {output_tokens}\n")

# === Calculate Total Cost ===
input_cost = (total_input_tokens / 1000) * INPUT_TOKEN_COST
output_cost = (total_output_tokens / 1000) * OUTPUT_TOKEN_COST
total_cost = input_cost + output_cost

# === Summary ===
print("\n===== 📊 Token Usage Summary =====")
print(f"🔹 Total Input Tokens: {total_input_tokens}")
print(f"🔹 Total Output Tokens: {total_output_tokens}")
print(f"💰 Total Cost: ${total_cost:.4f} (Input: ${input_cost:.4f}, Output: ${output_cost:.4f})")

# === Save Token Usage Summary ===
usage_summary = {
    "total_input_tokens": total_input_tokens,
    "total_output_tokens": total_output_tokens,
    "input_cost": round(input_cost, 4),
    "output_cost": round(output_cost, 4),
    "total_cost": round(total_cost, 4)
}

with open(USAGE_FILE, "w") as usage_file:
    json.dump(usage_summary, usage_file, indent=4)

print(f"📄 Usage summary saved to {USAGE_FILE}")
