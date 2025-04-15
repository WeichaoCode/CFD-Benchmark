import re
import pandas as pd
from datetime import datetime
import os


def call_create_table(llm_model, prompt_json):
    # === Config ===
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # PDE_Benchmark root
    LOG_FILE_PATH = os.path.join(ROOT_DIR, f'compare/comparison_results_{llm_model}_{prompt_json}.log')
    CSV_FILE_PATH = os.path.join(ROOT_DIR, f'table/{llm_model}_{prompt_json}_extracted_results_table_{timestamp}.csv')
    log_file_path = LOG_FILE_PATH
    output_csv_path = CSV_FILE_PATH

    # === Regex pattern for results line ===
    pattern = re.compile(
        r"INFO - ([\w_.\-]+): MSE=([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?), "
        r"MAE=([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?), "
        r"RMSE=([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?), "
        r"Cosine=([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?), "
        r"R2=([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
    )

    # === Extract Data ===
    results = []
    with open(log_file_path, 'r') as file:
        for line in file:
            match = pattern.search(line)
            if match:
                results.append({
                    'Filename': match.group(1),
                    'MSE': float(match.group(2)),
                    'MAE': float(match.group(3)),
                    'RMSE': float(match.group(4)),
                    'Cosine Similarity': float(match.group(5)),
                    'R-squared': float(match.group(6)),
                })

    # === Create DataFrame ===
    df = pd.DataFrame(results)

    # Format all float columns to scientific notation with 3 significant digits
    for col in ['MSE', 'MAE', 'RMSE', 'Cosine Similarity', 'R-squared']:
        df[col] = df[col].apply(lambda x: f"{x:.3e}")

    # === Save to CSV ===
    df.to_csv(output_csv_path, index=False)

    # === Display all rows/columns ===
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print("✅ Table saved to:", output_csv_path)
        print(df)


call_create_table("gpt-4o", "prompts")
call_create_table("o1-mini", "prompts")
call_create_table("sonnet-35", "prompts")
call_create_table("haiku", "prompts")


