import boto3
import json
import os
import re

# Define input JSON and output folder
input_json = "/opt/CFD-Benchmark/MMS/data/cfd_prompt.json"  # JSON containing prompts
output_folder = "/opt/CFD-Benchmark/MMS/generated_code"  # Folder to save Python code files

# Ensure the output directory exists
os.makedirs(output_folder, exist_ok=True)

# Initialize AWS Bedrock client
bedrock_runtime = boto3.client("bedrock-runtime", region_name="us-west-2")

# Define Sonnet-3.5 profile Inference Profile ARN
inference_profile_arn = "arn:aws:bedrock:us-west-2:991404956194:application-inference-profile/56i8iq1vib3e"

# Load CFD problem JSON
with open(input_json, "r") as file:
    data = json.load(file)

# Process each prompt
for problem in data["prompts"][:5]:
    name = problem["name"].replace(" ", "_")  # Replace spaces with underscores for valid filenames
    user_prompt = problem["prompt"]

    print(f"Generating code for: {name}...")

    # Construct API request
    request_body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 8000,
        "temperature": 0.7,
        "messages": [
            {"role": "user", "content": user_prompt}
        ]
    }

    try:
        # Invoke AWS Bedrock API
        response = bedrock_runtime.invoke_model(
            modelId=inference_profile_arn,
            body=json.dumps(request_body),
            contentType="application/json",
            accept="application/json"
        )

        # Parse the response
        response_body = json.loads(response["body"].read().decode("utf-8"))

        # Extract model response
        if "content" in response_body:
            model_response = response_body["content"][0]["text"]
        else:
            model_response = "Error: No response received from model."

        # Extract Python code from the response using regex
        code_match = re.findall(r"```python(.*?)```", model_response, re.DOTALL)
        if code_match:
            extracted_code = code_match[0].strip()  # Take the first code block
        else:
            extracted_code = "# No valid Python code extracted"

    except Exception as e:
        extracted_code = f"# Error: {str(e)}"

    # Save extracted code to a Python file
    file_path = os.path.join(output_folder, f"{name}.py")
    with open(file_path, "w") as code_file:
        code_file.write(extracted_code)

    print(f"Saved: {file_path}")

print("\nAll code files have been generated successfully.")
