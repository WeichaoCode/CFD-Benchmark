import os

def generate_tree_limited_depth(root_dir, max_depth=2, current_depth=0, prefix=""):
    tree_str = ""
    if current_depth >= max_depth:
        return tree_str

    try:
        entries = sorted(os.listdir(root_dir))
    except Exception:
        return ""

    pointers = ["├── "] * (len(entries) - 1) + ["└── "] if entries else []

    for pointer, name in zip(pointers, entries):
        path = os.path.join(root_dir, name)
        tree_str += f"{prefix}{pointer}{name}\n"

        if os.path.isdir(path):
            extension = "│   " if pointer == "├── " else "    "
            tree_str += generate_tree_limited_depth(path, max_depth, current_depth + 1, prefix + extension)

    return tree_str

if __name__ == "__main__":
    root = "."  # 你可以改为任何目录
    tree_output = generate_tree_limited_depth(root, max_depth=2)
    print(tree_output)

    # 保存为 Markdown
    with open("folder_structure.md", "w") as f:
        f.write("```\n")
        f.write(tree_output)
        f.write("```\n")
