# Downloads the prepared train/test splits from the Hugging Face Hub, trains and tunes
# several candidate models with MLflow experiment tracking, selects the best model by
# F1-score, and pushes the fitted pipeline to a Hugging Face Model Hub repo.

import os
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn
from huggingface_hub import HfApi, create_repo, hf_hub_download
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

HF_USERNAME = os.getenv("HF_USERNAME", "pavansainath")
DATASET_REPO_ID = f"{HF_USERNAME}/tourism-wellness-dataset"
MODEL_REPO_ID = f"{HF_USERNAME}/tourism-wellness-model"
HF_TOKEN = os.getenv("HF_TOKEN")

train_path = hf_hub_download(repo_id=DATASET_REPO_ID, repo_type="dataset", filename="train.csv", token=HF_TOKEN)
test_path = hf_hub_download(repo_id=DATASET_REPO_ID, repo_type="dataset", filename="test.csv", token=HF_TOKEN)
train, test = pd.read_csv(train_path), pd.read_csv(test_path)

target = "ProdTaken"
Xtrain, ytrain = train.drop(columns=[target]), train[target]
Xtest, ytest = test.drop(columns=[target]), test[target]

numeric_features = Xtrain.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = Xtrain.select_dtypes(include=["object"]).columns.tolist()

preprocessor = ColumnTransformer(transformers=[
    ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric_features),
    ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical_features),
])

models_and_params = {
    "LogisticRegression": (LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
                            {"classifier__C": [0.01, 0.1, 1, 10]}),
    "DecisionTree": (DecisionTreeClassifier(class_weight="balanced", random_state=42),
                      {"classifier__max_depth": [4, 6, 8, None], "classifier__min_samples_leaf": [1, 5, 10]}),
    "RandomForest": (RandomForestClassifier(class_weight="balanced", random_state=42),
                      {"classifier__n_estimators": [200, 400], "classifier__max_depth": [None, 10, 20]}),
    "GradientBoosting": (GradientBoostingClassifier(random_state=42),
                          {"classifier__n_estimators": [100, 200], "classifier__learning_rate": [0.05, 0.1]}),
}

mlflow.set_experiment("tourism-wellness-package-prediction")

best_score, best_model_name, best_pipeline = -1, None, None
for name, (estimator, param_grid) in models_and_params.items():
    pipe = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", estimator)])
    with mlflow.start_run(run_name=name):
        grid = GridSearchCV(pipe, param_grid, scoring="f1", cv=5, n_jobs=-1)
        grid.fit(Xtrain, ytrain)
        best_est = grid.best_estimator_

        preds = best_est.predict(Xtest)
        proba = best_est.predict_proba(Xtest)[:, 1]
        metrics = {
            "accuracy": accuracy_score(ytest, preds),
            "precision": precision_score(ytest, preds),
            "recall": recall_score(ytest, preds),
            "f1": f1_score(ytest, preds),
            "roc_auc": roc_auc_score(ytest, proba),
        }
        mlflow.log_params(grid.best_params_)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(best_est, name="model",serialization_format="cloudpickle")
        print(name, metrics)

        if metrics["f1"] > best_score:
            best_score, best_model_name, best_pipeline = metrics["f1"], name, best_est

print(f"Best model: {best_model_name} (F1={best_score:.3f})")

os.makedirs("tourism_project/model_building/artifacts", exist_ok=True)
model_path = "tourism_project/model_building/artifacts/best_model.joblib"
joblib.dump(best_pipeline, model_path)

api = HfApi(token=HF_TOKEN)
try:
    api.repo_info(repo_id=MODEL_REPO_ID, repo_type="model")
except Exception:
    create_repo(repo_id=MODEL_REPO_ID, repo_type="model", private=False, token=HF_TOKEN)
api.upload_file(path_or_fileobj=model_path, path_in_repo="best_model.joblib",
                 repo_id=MODEL_REPO_ID, repo_type="model", token=HF_TOKEN)
print(f"Best model pushed to https://huggingface.co/{MODEL_REPO_ID}")
