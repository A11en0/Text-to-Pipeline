import os
import json
import zipfile
from huggingface_hub import hf_hub_download

REPO_ID = "momo006/PARROT"
SAVE_DIR = "./"
os.makedirs(SAVE_DIR, exist_ok=True)

def process_split(split):
    print(f"\n🔽 Processing split: {split}")

    # Download benchmark.jsonl
    jsonl_path = hf_hub_download(repo_id=REPO_ID, repo_type="dataset", filename=f"{split}/benchmark.jsonl")
    tasks = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            item = json.loads(line)

            # Add status field
            item["status"] = "success"

            # Parse transform_chain_str
            if "transform_chain_str" in item:
                try:
                    item["transform_chain"] = json.loads(item["transform_chain_str"])
                except Exception as e:
                    print(f"❌ Failed to parse transform_chain_str, task_id={item.get('task_id')}: {e}")
                    item["transform_chain"] = []

            tasks.append(item)
    
    # Convert to JSON format
    json_data = {"tasks": tasks}

    # Create target directory
    split_dir = os.path.join(SAVE_DIR, split)
    os.makedirs(split_dir, exist_ok=True)

    # Save as JSON file (inside the split directory)
    json_output_path = os.path.join(split_dir, f"benchmark.json")
    with open(json_output_path, 'w') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    print(f"✅ Saved JSON file: {json_output_path}")

    # Download and extract csv_files.zip
    zip_path = hf_hub_download(repo_id=REPO_ID, repo_type="dataset", filename=f"{split}/csv_files.zip")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(split_dir)
    print(f"✅ Extracted CSV files to: {split_dir}")

# Process all splits
for split in ["test"]:
    process_split(split)
