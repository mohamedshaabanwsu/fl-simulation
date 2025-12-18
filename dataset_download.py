import kagglehub
import os
import shutil

# Download to kagglehub cache
dataset_path = kagglehub.dataset_download("apollo2506/eurosat-dataset")

# Create ./input (relative to current working directory) if it doesn't exist
target_dir = os.path.join(os.getcwd(), "input")
os.makedirs(target_dir, exist_ok=True)

# Move everything from the downloaded path into ./input
for name in os.listdir(dataset_path):
    src = os.path.join(dataset_path, name)
    dst = os.path.join(target_dir, name)
    if os.path.isdir(src):
        # Move directory tree
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.move(src, dst)
    else:
        shutil.move(src, dst)

print("Files now in:", target_dir)
