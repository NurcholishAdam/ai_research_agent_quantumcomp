import os
import re
from pathlib import Path

# Define valid SPDX header
SPDX_HEADER = "# SPDX-License-Identifier: Apache-2.0"

# File extensions to check
VALID_EXTENSIONS = [".py", ".ipynb", ".qml", ".qasm"]

def check_spdx_headers(repo_path: str):
    missing_spdx = []
    for root, _, files in os.walk(repo_path):
        for file in files:
            ext = Path(file).suffix
            if ext in VALID_EXTENSIONS:
                file_path = os.path.join(root, file)
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    first_line = f.readline().strip()
                    if SPDX_HEADER not in first_line:
                        missing_spdx.append(file_path)
    return missing_spdx

def main():
    repo_path = "."  # Current directory
    print("🔍 Checking SPDX headers...")
    missing = check_spdx_headers(repo_path)
    if missing:
        print(f"❌ Missing SPDX headers in {len(missing)} files:")
        for path in missing:
            print(f" - {path}")
    else:
        print("✅ All source files contain valid SPDX headers.")

if __name__ == "__main__":
    main()
