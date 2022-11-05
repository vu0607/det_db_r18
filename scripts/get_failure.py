import os
import shutil
from pathlib import Path

if __name__ == '__main__':
    root_dir = Path("test_result")
    true_dir = root_dir / "true"
    false_dir = root_dir / "false"
    fail_dir = root_dir / "fail"

    if os.path.isdir(fail_dir):
        shutil.rmtree(fail_dir)
    os.makedirs(fail_dir)

    true_file = os.listdir(true_dir)
    false_file = os.listdir(false_dir)

    for file in false_file:
        if file not in true_file:
            shutil.copy2(false_dir / file, fail_dir / file)
    print("yeah yeah")
