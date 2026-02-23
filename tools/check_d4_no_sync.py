#!/usr/bin/env python3
"""
CI check: GPU D4 augmentation must not call .item() in apply_d4_augment_batch_gpu.
.item() on GPU tensors causes sync. Run: python tools/check_d4_no_sync.py
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    path = Path("src/network/d4_augmentation_gpu.py")
    content = path.read_text()
    match = re.search(
        r"def apply_d4_augment_batch_gpu\(.*?\)(.*?)(?=\n\ndef |\nclass |\Z)",
        content,
        re.DOTALL,
    )
    if not match:
        print("WARN: Could not find apply_d4_augment_batch_gpu")
        return 0
    func_body = match.group(1)
    if re.search(r"\w+\.item\s*\(\s*\)", func_body):
        print("ERROR: apply_d4_augment_batch_gpu contains .item() call - causes GPU sync")
        for i, line in enumerate(func_body.splitlines(), 1):
            if re.search(r"\w+\.item\s*\(\s*\)", line):
                print(f"  Line: {line.strip()}")
        return 1
    print("OK: No .item() in apply_d4_augment_batch_gpu")
    return 0


if __name__ == "__main__":
    sys.exit(main())
