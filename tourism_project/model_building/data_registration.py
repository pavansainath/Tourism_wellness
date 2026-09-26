# Registers (uploads) the raw tourism.csv file as a Dataset repository on the Hugging Face Hub.
# This script is executed locally for experimentation and by the "register-dataset" job
# in the GitHub Actions pipeline.

import os
from huggingface_hub import HfApi, create_repo

HF_USERNAME = os.getenv("HF_USERNAME", "pavansainath")
DATASET_REPO_ID = f"{HF_USERNAME}/tourism-wellness-dataset"
HF_TOKEN = os.getenv("HF_TOKEN")

api = HfApi(token=HF_TOKEN)

# Create the dataset repo if it does not already exist (idempotent)
try:
    api.repo_info(repo_id=DATASET_REPO_ID, repo_type="dataset")
    print(f"Dataset repo '{DATASET_REPO_ID}' already exists.")
except Exception:
    create_repo(repo_id=DATASET_REPO_ID, repo_type="dataset", private=False, token=HF_TOKEN)
    print(f"Dataset repo '{DATASET_REPO_ID}' created.")

api.upload_file(
    path_or_fileobj="tourism_project/data/tourism.csv",
    path_in_repo="tourism.csv",
    repo_id=DATASET_REPO_ID,
    repo_type="dataset",
    token=HF_TOKEN,
)
print(f"Raw dataset uploaded to https://huggingface.co/datasets/{DATASET_REPO_ID}")
