"""
Name: compare_output.py
Author: Weichao Li
Time: 2025-03-25
Description:
    This script compares two folders containing `.npy` files and identifies:
    - Files that exist in both folders
    - Files only in folder A
    - Files only in folder B

    It is useful for checking consistency of saved NumPy results across experiments or outputs.
"""
import os

# === Paths ===
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # PDE_Benchmark root
# === Set your folder paths ===
folder_a = os.path.join(ROOT_DIR, 'results/ground_truth')  # Replace with your first folder path
folder_b = os.path.join(ROOT_DIR, 'results/prediction/o1-mini')  # Replace with your second folder path

# === Collect .npy filenames ===
files_a = {f for f in os.listdir(folder_a) if f.endswith('.npy')}
files_b = {f for f in os.listdir(folder_b) if f.endswith('.npy')}

# === Compare ===
common_files = sorted(files_a & files_b)
only_in_a = sorted(files_a - files_b)
only_in_b = sorted(files_b - files_a)

# === Report ===
print("\n✅ Files in BOTH folders:")
for f in common_files:
    print(f"  - {f}")

print("\n📁 Files ONLY in folder A:")
for f in only_in_a:
    print(f"  - {f}")

print("\n📁 Files ONLY in folder B:")
for f in only_in_b:
    print(f"  - {f}")

# === Summary ===
print("\n🧾 Summary:")
print(f"  Total in A: {len(files_a)}")
print(f"  Total in B: {len(files_b)}")
print(f"  Common:    {len(common_files)}")
print(f"  Only A:    {len(only_in_a)}")
print(f"  Only B:    {len(only_in_b)}")
