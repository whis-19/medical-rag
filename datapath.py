#!/usr/bin/env python3
"""
scan_dataset_structure_full.py

Scans a dataset directory (like OpenNeuro EEG datasets) recursively without any depth limit.

Outputs:
- Directory tree structure
- File extensions and counts per folder
- Example filenames
- Optional file sizes (in MB)

Saves everything into `dataset_structure.txt` for inspection before model fine-tuning.
"""

import os
from pathlib import Path
from collections import Counter

def scan_dataset(root_path="dataset", output_file="dataset_structure.txt", examples_per_ext=3, include_size=True):
    root = Path(root_path)
    if not root.exists():
        raise FileNotFoundError(f"❌ Path not found: {root}")

    total_files = 0
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"Dataset Structure Report for: {root}\n")
        f.write("=" * 100 + "\n\n")

        for dirpath, dirnames, filenames in os.walk(root):
            depth = dirpath.replace(str(root), "").count(os.sep)
            indent = "  " * depth
            rel_path = os.path.relpath(dirpath, root)
            if rel_path == ".":
                rel_path = root.name

            f.write(f"{indent}[DIR] {rel_path}\n")

            if filenames:
                total_files += len(filenames)
                exts = [Path(fn).suffix.lower() for fn in filenames]
                ext_counter = Counter(exts)

                for ext, count in ext_counter.items():
                    f.write(f"{indent}  ├── {ext or '(no extension)'}: {count} files\n")

                    # Example files for this extension
                    example_files = [fn for fn in filenames if fn.lower().endswith(ext)]
                    for ex in example_files[:examples_per_ext]:
                        file_path = Path(dirpath) / ex
                        if include_size:
                            try:
                                size_mb = file_path.stat().st_size / (1024 * 1024)
                                f.write(f"{indent}      • {ex}  ({size_mb:.2f} MB)\n")
                            except Exception:
                                f.write(f"{indent}      • {ex}  (size unavailable)\n")
                        else:
                            f.write(f"{indent}      • {ex}\n")

            f.write("\n")

        f.write("=" * 100 + f"\n\nTotal files scanned: {total_files}\n")

    print(f"✅ Full scan complete. Report saved to '{output_file}'. Total files: {total_files}")


if __name__ == "__main__":
    scan_dataset()