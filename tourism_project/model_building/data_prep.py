# Downloads the raw dataset from the Hugging Face Hub, cleans it, splits it into
# train/test sets, and re-uploads the splits to the same Hugging Face Dataset repo.

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from huggingface_hub import HfApi, hf_hub_download

HF_USERNAME = os.getenv("HF_USERNAME", "pavansainath")
DATASET_REPO_ID = f"{HF_USERNAME}/tourism-wellness-dataset"
HF_TOKEN = os.getenv("HF_TOKEN")

# 1. Download the raw data registered in the previous pipeline stage
raw_path = hf_hub_download(
    repo_id=DATASET_REPO_ID, repo_type="dataset", filename="tourism.csv", token=HF_TOKEN
)
df = pd.read_csv(raw_path)
df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")])

# 2. Clean
df["Gender"] = df["Gender"].replace({"Fe Male": "Female"})
df = df.drop(columns=["CustomerID"])

# 3. Split
X = df.drop(columns=["ProdTaken"])
y = df["ProdTaken"]
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

os.makedirs("tourism_project/data", exist_ok=True)
train = pd.concat([Xtrain, ytrain], axis=1)
test = pd.concat([Xtest, ytest], axis=1)
train.to_csv("tourism_project/data/train.csv", index=False)
test.to_csv("tourism_project/data/test.csv", index=False)

# 4. Upload the processed splits back to the Hugging Face Hub
api = HfApi(token=HF_TOKEN)
api.upload_file(path_or_fileobj="tourism_project/data/train.csv", path_in_repo="train.csv",
                 repo_id=DATASET_REPO_ID, repo_type="dataset", token=HF_TOKEN)
api.upload_file(path_or_fileobj="tourism_project/data/test.csv", path_in_repo="test.csv",
                 repo_id=DATASET_REPO_ID, repo_type="dataset", token=HF_TOKEN)

print(f"Data preparation complete. train.csv ({len(train)} rows) and "
      f"test.csv ({len(test)} rows) uploaded to https://huggingface.co/datasets/{DATASET_REPO_ID}")
