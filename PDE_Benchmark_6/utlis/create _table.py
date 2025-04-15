import pandas as pd
import re
import os

# Define the directory where generated solver scripts are stored
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # PDE_Benchmark root
REPORT_DIR = os.path.join(ROOT_DIR, 'report')


def extract_mse_from_log(log_file_path):
    """Extracts problem names and Mean Squared Errors (MSE) from a log file."""
    # Read log file
    with open(log_file_path, "r") as file:
        log_data = file.readlines()

    # Extract problem names and MSE values using regex
    data = []
    pattern = re.compile(r"current problem is (\S+), Mean Squared Error: ([\d\.]+)")

    for line in log_data:
        match = pattern.search(line)
        if match:
            problem = match.group(1)
            mse = float(match.group(2))
            data.append((problem, mse))

    # Create a DataFrame
    df = pd.DataFrame(data, columns=["Problem", "Mean Squared Error"])

    return df


# Example usage
log_file_path = os.path.join(REPORT_DIR, 'compare.log')  # Change this to the actual log file path
df = extract_mse_from_log(log_file_path)

# Display the table
print(df)

# Save the table as a CSV file
table_path = os.path.join(REPORT_DIR, 'mse_results.csv')
df.to_csv(table_path, index=False)
