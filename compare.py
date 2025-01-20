import subprocess
import re
import numpy as np
import datetime

def extract_numbers(output):
    """Extract numerical values (integers and floats) from output string."""
    return np.array([float(num) for num in re.findall(r"[-+]?\d*\.\d+|\d+", output)])


def mean_absolute_percentage_error(y_true, y_pred):
    """Computes MAPE: Mean Absolute Percentage Error."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100  # Percentage error


def mean_squared_error(y_true, y_pred):
    """Computes Mean Squared Error (MSE)."""
    return np.mean((y_true - y_pred) ** 2)


def compare_code_output_loss(file1, file2):
    try:
        # Run both scripts and capture outputs
        result1 = subprocess.run(["python", file1], capture_output=True, text=True, timeout=5)
        result2 = subprocess.run(["python", file2], capture_output=True, text=True, timeout=5)

        # Extract numerical values
        numbers1 = extract_numbers(result1.stdout)
        numbers2 = extract_numbers(result2.stdout)

        # Ensure both outputs contain numbers
        if len(numbers1) == 0 or len(numbers2) == 0:
            return "Error: One or both outputs contain no numeric values."

        # Ensure both outputs are the same length
        if len(numbers1) != len(numbers2):
            return f"Error: Output lengths differ (Response: {len(numbers1)}, Instruction: {len(numbers2)})"

        # Compute loss functions
        mape = mean_absolute_percentage_error(numbers1, numbers2)
        mse = mean_squared_error(numbers1, numbers2)

        return {
            "MAPE (%)": round(mape, 4),
            "MSE": round(mse, 6),
            "Same Output": mape < 1.0  # Consider outputs "same" if relative error is < 1%
        }

    except Exception as e:
        return str(e)


# Example usage
file1 = "response/3.py"
file2 = "instruction/3.py"

# Extract the key dynamically (assumes filenames follow "X.py" format)
key_match = re.search(r"(\d+)\.py", file1)  # Extracts '3' from "3.py"
if key_match:
    key = key_match.group(1)
else:
    raise ValueError("Invalid file format. Expected 'X.py' structure.")

# Generate log entry
result = compare_code_output_loss(file1, file2)
log_entry = f"compare key {key}, {result}"

# Log file path
log_file = "output_log.txt"

# Read existing log data
log_lines = []
existing_keys = {}

try:
    with open(log_file, "r") as log:
        for line in log:
            match = re.search(r"compare key (\d+),", line)
            if match:
                existing_keys[match.group(1)] = line.strip()
            log_lines.append(line.strip())
except FileNotFoundError:
    pass  # Log file doesn't exist yet, will be created

# Update log (rewrite existing key entry or append new)
if key in existing_keys:
    log_lines = [log_entry if f"compare key {key}," in line else line for line in log_lines]
else:
    log_lines.append(log_entry)

# Write updated log back to file
with open(log_file, "w") as log:
    for line in log_lines:
        log.write(line + "\n")


print(result)
