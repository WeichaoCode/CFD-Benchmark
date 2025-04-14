import os
import shutil
import glob


def make_clean(root_dir):
    """
    Delete all generated folders and `.npy` files to return to a clean state.
    Similar to `make clean` in C/C++ projects.
    """
    # Folders to remove
    generated_dirs = [
        "solver",
        "results/prediction",
        "report",
        "compare",
        "compare_images",
        "table",
        "image"
    ]

    print("🧹 Cleaning generated folders...\n")

    for subdir in generated_dirs:
        full_path = os.path.join(root_dir, subdir)
        if os.path.exists(full_path):
            try:
                shutil.rmtree(full_path)
                print(f"✅ Removed directory: {full_path}")
            except Exception as e:
                print(f"❌ Failed to remove {full_path}: {e}")
        else:
            print(f"ℹ️ Skipped (not found): {full_path}")

    # Delete all .npy files in root_dir (non-recursive)
    npy_files = glob.glob(os.path.join(root_dir, "*.npy"))
    for file_path in npy_files:
        try:
            os.remove(file_path)
            print(f"🗑️ Deleted file: {file_path}")
        except Exception as e:
            print(f"❌ Failed to delete {file_path}: {e}")

    print("\n✨ Done. Project workspace is clean.")


make_clean("/opt/CFD-Benchmark/PDE_Benchmark")
