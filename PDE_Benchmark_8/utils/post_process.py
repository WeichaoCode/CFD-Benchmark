import os
import re


def call_post_process(llm_model, prompt_json):
    # === Paths ===
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # PDE_Benchmark root
    # Set your folder path
    folder_path = os.path.join(ROOT_DIR, f'solver/{llm_model}/{prompt_json}')  # <-- Replace with your actual folder
    save_dir = os.path.join(ROOT_DIR, f'results/prediction/{llm_model}/{prompt_json}')
    # Process each Python file in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".py"):
            py_path = os.path.join(folder_path, filename)
            base_name = os.path.splitext(filename)[0]  # e.g., "burgers_solver"

            with open(py_path, "r") as f:
                code = f.read()

            # Pattern to match np.save('somefile.npy', variable)
            pattern = r"np\.save\((['\"])(.+?\.npy)\1\s*,\s*(\w+)\s*\)"

            def replacer(match):
                var_name = match.group(3)  # u, v, p, etc.
                new_filename = f"{var_name}_{base_name}.npy"
                new_path = os.path.join(save_dir, new_filename).replace("\\", "/")  # make sure it's Unix-style for code
                return f"np.save('{new_path}', {var_name})"

            # Apply the replacement
            new_code, count = re.subn(pattern, replacer, code)

            if count > 0:
                with open(py_path, "w") as f:
                    f.write(new_code)
                print(f"✅ Updated {filename}: replaced {count} save path(s)")
            else:
                print(f"ℹ️ No np.save calls updated in {filename}")


call_post_process('gpt-4o', 'prompts_both_instructions')
#
# call_post_process('gpt-4o', 'prompts_instruction_1')
#
# call_post_process('gpt-4o', 'prompts_instruction_2')

# call_post_process('gpt-4o', 'prompts_no_instruction')
