import os
import json
import re
from openai import OpenAI

# API key and OpenAI client setup
api_key = "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"
client = OpenAI(api_key=api_key)

# Define input JSON and output folder
input_json = "/opt/CFD-Benchmark/MMS/data_1/cfd_prompts.json"
output_folder = "/opt/CFD-Benchmark/MMS/generated_code/gpt-4/2"

# Ensure the output directory exists
os.makedirs(output_folder, exist_ok=True)

# Load CFD problem JSON
with open(input_json, "r") as file:
    data = json.load(file)

# Token usage and cost tracking
total_input_tokens = 0
total_output_tokens = 0

# Pricing (adjust if necessary)
input_token_price_per_1k = 0.03  # $ per 1,000 input tokens
output_token_price_per_1k = 0.06  # $ per 1,000 output tokens

# Process each prompt
for problem in data["prompts"][1:2]:
    key = problem["key"]
    name = problem["name"].replace(" ", "_")
    user_prompt = problem["prompt"]

    print(f"Generating code for key {key}: {name}...")

    try:
        # Call OpenAI Chat API
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert in CFD and Python."},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=7000,
            temperature=0.7
        )

        # Extract model response
        model_response = response.choices[0].message.content.strip()

        # Extract Python code from response
        code_match = re.findall(r"```python(.*?)```", model_response, re.DOTALL)
        extracted_code = code_match[0].strip() if code_match else "# No valid Python code extracted"

        # Track token usage
        usage = response.usage
        input_tokens = usage.prompt_tokens
        output_tokens = usage.completion_tokens

        total_input_tokens += input_tokens
        total_output_tokens += output_tokens

    except Exception as e:
        extracted_code = f"# Error: {str(e)}"
        input_tokens = 0
        output_tokens = 0

    # Save code to file
    file_path = os.path.join(output_folder, f"{name}_{key}.py")
    with open(file_path, "w") as code_file:
        code_file.write(extracted_code)

    print(f"Saved: {file_path}")
    print(f"Input Tokens: {input_tokens}, Output Tokens: {output_tokens}")

# Calculate total cost
input_cost = (total_input_tokens / 1000) * input_token_price_per_1k
output_cost = (total_output_tokens / 1000) * output_token_price_per_1k
total_cost = input_cost + output_cost

# Summary
print("\n===== Token Usage Summary =====")
print(f"Total Input Tokens Used: {total_input_tokens}")
print(f"Total Output Tokens Generated: {total_output_tokens}")
print(f"Total Cost: ${total_cost:.4f} (Input: ${input_cost:.4f}, Output: ${output_cost:.4f})")

# Save token usage and cost details to a JSON file
usage_summary = {
    "total_input_tokens": total_input_tokens,
    "total_output_tokens": total_output_tokens,
    "input_cost": round(input_cost, 4),
    "output_cost": round(output_cost, 4),
    "total_cost": round(total_cost, 4)
}

with open(os.path.join(output_folder, "token_usage_summary.json"), "w") as usage_file:
    json.dump(usage_summary, usage_file, indent=4)

print(f"Usage summary saved to {os.path.join(output_folder, 'token_usage_summary.json')}")
