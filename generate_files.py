import os
import ast

project_path = "."  # فولدر المشروع

packages = set()

# =========================
# Extract imports from files
# =========================
for root, dirs, files in os.walk(project_path):
    for file in files:
        if file.endswith(".py"):
            with open(os.path.join(root, file), "r", encoding="utf-8") as f:
                try:
                    tree = ast.parse(f.read())
                except:
                    continue

                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for n in node.names:
                            packages.add(n.name.split(".")[0])
                    elif isinstance(node, ast.ImportFrom):
                        if node.module:
                            packages.add(node.module.split(".")[0])

# =========================
# Remove built-in modules
# =========================
ignore = {
    "os", "sys", "json", "math", "time", "pathlib",
    "typing", "collections", "itertools", "ast"
}

packages = [p for p in packages if p not in ignore]

# =========================
# Save requirements.txt
# =========================
with open("requirements.txt", "w") as f:
    for p in sorted(packages):
        f.write(p + "\n")

print("[OK] requirements.txt generated automatically")