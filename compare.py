import subprocess
import re
import numpy as np


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
result = compare_code_output_loss("response.py", "instruction.py")
print(result)

