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
        # Run both scripts and save the output arrays to temporary files
        subprocess.run(["python", file1], check=True, timeout=100)
        subprocess.run(["python", file2], check=True, timeout=100)

        # Load the u and v arrays from both script outputs
        u1 = np.load("u_true.npy")
        v1 = np.load("v_true.npy")
        u2 = np.load("u_pred.npy")
        v2 = np.load("v_pred.npy")

        # Compute differences (Mean Squared Error)
        u_loss = np.mean((u1 - u2) ** 2)
        v_loss = np.mean((v1 - v2) ** 2)

        print(f"Loss between u arrays (MSE): {u_loss:.6f}")
        print(f"Loss between v arrays (MSE): {v_loss:.6f}")

        # Compute absolute differences for debugging
        max_u_diff = np.max(np.abs(u1 - u2))
        max_v_diff = np.max(np.abs(v1 - v2))
        print(f"Max absolute difference in u: {max_u_diff:.6f}")
        print(f"Max absolute difference in v: {max_v_diff:.6f}")

        return u_loss, v_loss

    except Exception as e:
        print(f"Error comparing outputs: {e}")
        return None, None


# Example usage
file1 = "response/9.py"
file2 = "instruction/9.py"

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
