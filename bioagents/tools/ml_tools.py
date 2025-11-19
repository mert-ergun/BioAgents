"""Machine Learning tools for designing and executing ML systems in bioinformatics."""

import json
import os
import tempfile
from pathlib import Path

from langchain_core.tools import tool


@tool
def design_ml_pipeline(
    task_description: str,
    data_description: str,
    features: str,
    target: str,
    algorithm: str = "auto",
) -> str:
    """
    Design a machine learning pipeline for a bioinformatics task.

    This tool generates a complete Python ML pipeline including:
    - Data loading and preprocessing
    - Feature engineering
    - Model training
    - Evaluation metrics
    - Visualization

    Args:
        task_description: Description of the ML task (e.g., "protein classification",
                          "gene expression prediction")
        data_description: Description of the data format (e.g., "CSV file with columns:
                          sequence, length, gc_content, label")
        features: Comma-separated list of feature column names
        target: Name of the target variable column
        algorithm: ML algorithm to use. Options: "auto" (default), "random_forest",
                   "xgboost", "logistic_regression", "svm", "neural_network"

    Returns:
        A Python code string containing the complete ML pipeline that can be executed
    """
    try:
        feature_list = [f.strip() for f in features.split(",")]

        algorithm_code = _generate_algorithm_code(algorithm, feature_list, target)

        pipeline = f'''"""
ML Pipeline for: {task_description}
Data: {data_description}
Features: {features}
Target: {target}
Algorithm: {algorithm}
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, roc_auc_score, roc_curve
)
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

def load_and_preprocess_data(data_path):
    """Load and preprocess the data."""
    print("Loading data...")
    df = pd.read_csv(data_path)

    print(f"Dataset shape: {{df.shape}}")
    print(f"\\nFirst few rows:")
    print(df.head())

    print(f"\\nDataset info:")
    print(df.info())

    print(f"\\nMissing values:")
    print(df.isnull().sum())

    # Handle missing values
    if df.isnull().any().any():
        print("\\nHandling missing values...")
        df = df.fillna(df.mean(numeric_only=True))

    return df

def prepare_features(df):
    """Prepare features and target."""
    feature_cols = {feature_list}
    target_col = '{target}'

    print(f"\\nFeatures: {{feature_cols}}")
    print(f"Target: {{target_col}}")

    X = df[feature_cols]
    y = df[target_col]

    # Check if target is categorical
    if y.dtype == 'object' or len(np.unique(y)) < 20:
        print(f"\\nTarget variable distribution:")
        print(y.value_counts())

        # Encode if necessary
        if y.dtype == 'object':
            le = LabelEncoder()
            y = le.fit_transform(y)
            print(f"\\nEncoded target classes: {{le.classes_}}")

    # Feature scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=feature_cols)

    return X_scaled, y, scaler

def split_data(X, y, test_size=0.2, random_state=42):
    """Split data into train and test sets."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"\\nTrain set size: {{X_train.shape[0]}}")
    print(f"Test set size: {{X_test.shape[0]}}")

    return X_train, X_test, y_train, y_test

{algorithm_code}

def evaluate_model(model, X_test, y_test, y_pred):
    """Evaluate model performance."""
    print("\\n" + "="*50)
    print("MODEL EVALUATION")
    print("="*50)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"\\nAccuracy: {{accuracy:.4f}}")

    print(f"\\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
    print("\\nConfusion matrix saved as 'confusion_matrix.png'")

    # ROC curve (for binary classification)
    if len(np.unique(y_test)) == 2:
        try:
            y_proba = model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_proba)
            print(f"\\nAUC-ROC Score: {{auc:.4f}}")

            fpr, tpr, _ = roc_curve(y_test, y_proba)
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {{auc:.4f}})')
            plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve')
            plt.legend()
            plt.savefig('roc_curve.png', dpi=150, bbox_inches='tight')
            print("ROC curve saved as 'roc_curve.png'")
        except Exception as e:
            print(f"Could not generate ROC curve: {{e}}")

    return accuracy

def plot_feature_importance(model, feature_names):
    """Plot feature importance if available."""
    try:
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_).mean(axis=0) if model.coef_.ndim > 1 else np.abs(model.coef_)
        else:
            print("\\nModel does not support feature importance")
            return

        indices = np.argsort(importances)[::-1]

        plt.figure(figsize=(10, 6))
        plt.bar(range(len(importances)), importances[indices])
        plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45, ha='right')
        plt.title('Feature Importance')
        plt.xlabel('Features')
        plt.ylabel('Importance')
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
        print("\\nFeature importance plot saved as 'feature_importance.png'")

        print("\\nTop features:")
        for i in indices:
            print(f"  {{feature_names[i]}}: {{importances[i]:.4f}}")

    except Exception as e:
        print(f"Could not plot feature importance: {{e}}")

def main(data_path='data.csv'):
    """Main pipeline execution."""
    print("="*50)
    print(f"ML Pipeline: {task_description}")
    print("="*50)

    # Load and preprocess
    df = load_and_preprocess_data(data_path)

    # Prepare features
    X, y, scaler = prepare_features(df)

    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Train model
    model = train_model(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate
    accuracy = evaluate_model(model, X_test, y_test, y_pred)

    # Feature importance
    plot_feature_importance(model, list(X.columns))

    print("\\n" + "="*50)
    print("Pipeline completed successfully!")
    print("="*50)

    return model, scaler, accuracy

if __name__ == "__main__":
    # Execute pipeline
    model, scaler, accuracy = main()
    print(f"\\nFinal Accuracy: {{accuracy:.4f}}")
'''

        return pipeline

    except Exception as e:
        return f"Error designing ML pipeline: {e!s}"


def _generate_algorithm_code(algorithm: str, features: list, target: str) -> str:
    """Generate algorithm-specific training code."""

    if algorithm == "auto" or algorithm == "random_forest":
        return '''
def train_model(X_train, y_train):
    """Train Random Forest model."""
    from sklearn.ensemble import RandomForestClassifier

    print("\\nTraining Random Forest model...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )

    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Mean CV score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

    # Train final model
    model.fit(X_train, y_train)
    print("Model training completed!")

    return model
'''

    elif algorithm == "xgboost":
        return '''
def train_model(X_train, y_train):
    """Train XGBoost model."""
    try:
        import xgboost as xgb
    except ImportError:
        print("XGBoost not installed. Install with: pip install xgboost")
        from sklearn.ensemble import GradientBoostingClassifier
        print("Using GradientBoostingClassifier as fallback...")
        model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        return model

    print("\\nTraining XGBoost model...")
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1
    )

    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Mean CV score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

    # Train final model
    model.fit(X_train, y_train)
    print("Model training completed!")

    return model
'''

    elif algorithm == "logistic_regression":
        return '''
def train_model(X_train, y_train):
    """Train Logistic Regression model."""
    from sklearn.linear_model import LogisticRegression

    print("\\nTraining Logistic Regression model...")
    model = LogisticRegression(
        max_iter=1000,
        random_state=42,
        n_jobs=-1
    )

    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Mean CV score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

    # Train final model
    model.fit(X_train, y_train)
    print("Model training completed!")

    return model
'''

    elif algorithm == "svm":
        return '''
def train_model(X_train, y_train):
    """Train SVM model."""
    from sklearn.svm import SVC

    print("\\nTraining SVM model...")
    model = SVC(
        kernel='rbf',
        probability=True,
        random_state=42
    )

    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Mean CV score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

    # Train final model
    model.fit(X_train, y_train)
    print("Model training completed!")

    return model
'''

    elif algorithm == "neural_network":
        return '''
def train_model(X_train, y_train):
    """Train Neural Network model."""
    from sklearn.neural_network import MLPClassifier

    print("\\nTraining Neural Network model...")
    model = MLPClassifier(
        hidden_layer_sizes=(100, 50),
        activation='relu',
        max_iter=500,
        random_state=42
    )

    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Mean CV score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

    # Train final model
    model.fit(X_train, y_train)
    print("Model training completed!")

    return model
'''

    else:
        return _generate_algorithm_code("random_forest", features, target)


@tool
def execute_ml_code(
    code: str,
    data_csv: str | None = None,
    additional_files: str | None = None,
) -> str:
    """
    Execute ML code in a sandboxed Docker environment.

    This tool safely runs Python ML code in an isolated Docker container with
    pre-installed scientific computing libraries (numpy, pandas, scikit-learn, etc.).

    Args:
        code: Python code to execute
        data_csv: Optional CSV data as string. Will be saved as 'data.csv' in the sandbox
        additional_files: Optional JSON string mapping filenames to content
                         (e.g., '{"config.json": "{\\"param\\": 1}"}')

    Returns:
        Execution results including stdout, stderr, and information about generated files
    """
    try:
        import docker
        from docker.errors import DockerException, ImageNotFound

        try:
            client = docker.from_env()
            client.ping()
        except DockerException as e:
            return (
                f"Error: Docker is not available. Please ensure Docker is running.\nDetails: {e!s}"
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            code_path = Path(tmpdir) / "ml_script.py"
            with Path(code_path).open("w") as f:
                f.write(code)

            if data_csv and data_csv.strip():
                data_path = Path(tmpdir) / "data.csv"
                with Path(data_path).open("w") as f:
                    f.write(data_csv)

            if additional_files and additional_files.strip():
                try:
                    files_dict = json.loads(additional_files)
                    for filename, content in files_dict.items():
                        file_path = Path(tmpdir) / filename
                        with Path(file_path).open("w") as f:
                            f.write(content)
                except json.JSONDecodeError as e:
                    return f"Error: Invalid JSON in additional_files: {e!s}"

            output_dir = Path(tmpdir) / "output"
            output_dir.mkdir(parents=True, exist_ok=True)

            image_name = "python:3.10-slim"
            try:
                client.images.get(image_name)
            except ImageNotFound:
                print(f"Pulling Docker image {image_name}...")
                client.images.pull(image_name)

            print("Starting Docker container for ML execution...")
            container = client.containers.run(
                image_name,
                command=[
                    "sh",
                    "-c",
                    "pip install --no-cache-dir --quiet numpy pandas scikit-learn matplotlib seaborn 2>&1 && python /workspace/ml_script.py",
                ],
                volumes={tmpdir: {"bind": "/workspace", "mode": "rw"}},
                working_dir="/workspace",
                detach=True,
                mem_limit="2g",
                cpu_quota=100000,
                remove=False,
            )

            try:
                result = container.wait(timeout=300)
                exit_code = result.get("StatusCode", -1)

                combined_logs = container.logs(stdout=True, stderr=True).decode(
                    "utf-8", errors="replace"
                )
                stdout = container.logs(stdout=True, stderr=False).decode("utf-8", errors="replace")
                stderr = container.logs(stdout=False, stderr=True).decode("utf-8", errors="replace")

                container.remove()

                generated_files = []
                for root, _, files in os.walk(tmpdir):
                    for file in files:
                        if file not in [
                            "ml_script.py",
                            "data.csv",
                            "run.sh",
                        ] and not file.startswith("."):
                            file_path = Path(root) / file
                            rel_path = file_path.relative_to(tmpdir)
                            file_size = file_path.stat().st_size
                            generated_files.append(f"{rel_path} ({file_size} bytes)")

                output = "=" * 60 + "\n"
                output += "ML CODE EXECUTION RESULTS\n"
                output += "=" * 60 + "\n\n"

                if exit_code == 0:
                    output += "✓ Execution completed successfully\n\n"
                else:
                    output += f"✗ Execution failed with exit code {exit_code}\n\n"

                # If both stdout and stderr are empty, show combined logs
                if not stdout.strip() and not stderr.strip() and combined_logs.strip():
                    output += "COMBINED OUTPUT:\n" + "-" * 60 + "\n"
                    output += combined_logs + "\n"
                else:
                    output += "STDOUT:\n" + "-" * 60 + "\n"
                    output += stdout if stdout.strip() else "(no output)\n"

                    if stderr.strip():
                        output += "\nSTDERR:\n" + "-" * 60 + "\n"
                        output += stderr + "\n"

                if generated_files:
                    output += "\nGENERATED FILES:\n" + "-" * 60 + "\n"
                    for file_info in generated_files:
                        output += f"  - {file_info}\n"
                    output += "\nNote: Files are saved in the temporary directory and will be cleaned up.\n"
                    output += (
                        "To persist files, modify the code to save them to a permanent location.\n"
                    )

                output += "\n" + "=" * 60

                return output

            except Exception as e:
                try:
                    container.stop(timeout=5)
                    container.remove()
                except Exception:
                    return f"Error stopping container: {e!s}"
                return f"Error during execution: {e!s}"

    except ImportError:
        return "Error: Docker Python package not installed. Install with: pip install docker"
    except Exception as e:
        return f"Error executing ML code: {e!s}"


@tool
def optimize_ml_pipeline(performance_issue: str) -> str:
    """
    Suggest optimizations for an existing ML pipeline.

    This tool analyzes ML code and suggests improvements based on performance issues
    or specific optimization strategies.

    Args:
        performance_issue: Description of the issue (e.g., "low accuracy", "overfitting",
                          "slow training")

    Returns:
        Suggested code improvements and explanation
    """
    try:
        suggestions = {
            "low accuracy": """
# Suggestions for improving accuracy:

1. **Hyperparameter Tuning with Grid Search**:
```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(
    model, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=2
)
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_
print(f"Best parameters: {grid_search.best_params_}")
```

2. **Feature Engineering**: Add polynomial features or interaction terms:
```python
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2, interaction_only=True)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)
```

3. **Ensemble Methods**: Combine multiple models:
```python
from sklearn.ensemble import VotingClassifier

ensemble = VotingClassifier(
    estimators=[
        ('rf', RandomForestClassifier()),
        ('xgb', XGBClassifier()),
        ('lr', LogisticRegression())
    ],
    voting='soft'
)
ensemble.fit(X_train, y_train)
```
""",
            "overfitting": """
# Suggestions for reducing overfitting:

1. **Regularization for Random Forest**:
```python
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,  # Reduce depth
    min_samples_split=10,  # Increase minimum samples
    min_samples_leaf=4,  # Increase minimum leaf samples
    max_features='sqrt',  # Limit features per split
    random_state=42
)
```

2. **Cross-validation with more folds**:
```python
from sklearn.model_selection import cross_validate

cv_results = cross_validate(
    model, X_train, y_train, cv=10,
    scoring=['accuracy', 'precision', 'recall'],
    return_train_score=True
)

print(f"Train accuracy: {cv_results['train_accuracy'].mean():.4f}")
print(f"Test accuracy: {cv_results['test_accuracy'].mean():.4f}")
# Look for gap between train and test scores
```

3. **Feature Selection**:
```python
from sklearn.feature_selection import SelectKBest, f_classif

selector = SelectKBest(f_classif, k=10)  # Select top 10 features
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)
```

4. **Early Stopping for XGBoost**:
```python
model = XGBClassifier(
    n_estimators=1000,
    early_stopping_rounds=10,
    eval_metric='logloss'
)
model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=False
)
```
""",
            "slow training": """
# Suggestions for faster training:

1. **Feature Selection** (reduce dimensionality):
```python
from sklearn.feature_selection import SelectKBest, mutual_info_classif

# Select top k features based on mutual information
k = min(20, X_train.shape[1])  # Select top 20 or all if fewer
selector = SelectKBest(mutual_info_classif, k=k)
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)
```

2. **Reduce Model Complexity**:
```python
# For Random Forest
model = RandomForestClassifier(
    n_estimators=50,  # Reduce number of trees
    max_depth=10,  # Limit tree depth
    n_jobs=-1,  # Use all CPU cores
    random_state=42
)

# For Neural Network
model = MLPClassifier(
    hidden_layer_sizes=(50,),  # Smaller network
    max_iter=200,  # Fewer iterations
    early_stopping=True,
    random_state=42
)
```

3. **Sample Data for Large Datasets**:
```python
from sklearn.model_selection import StratifiedShuffleSplit

# Use stratified sampling for faster training
splitter = StratifiedShuffleSplit(n_splits=1, train_size=0.5, random_state=42)
sample_idx, _ = next(splitter.split(X_train, y_train))
X_train_sample = X_train[sample_idx]
y_train_sample = y_train[sample_idx]
```
""",
            "unbalanced classes": """
# Suggestions for handling class imbalance:

1. **Class Weights**:
```python
from sklearn.utils.class_weight import compute_class_weight

# Compute balanced class weights
class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weight_dict = dict(enumerate(class_weights))

# Use in model
model = RandomForestClassifier(
    class_weight=class_weight_dict,
    random_state=42
)
```

2. **SMOTE (Synthetic Minority Over-sampling)**:
```python
from imblearn.over_sampling import SMOTE

# Install: pip install imbalanced-learn
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print(f"Original class distribution: {np.bincount(y_train)}")
print(f"Resampled class distribution: {np.bincount(y_train_resampled)}")
```

3. **Stratified Sampling**:
```python
# Ensure stratification in train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Use stratified CV
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X_train, y_train, cv=skf)
```
""",
        }

        issue_lower = performance_issue.lower()
        suggestion_text = ""

        for key, value in suggestions.items():
            if key in issue_lower:
                suggestion_text = value
                break

        if not suggestion_text:
            suggestion_text = """
# General Optimization Suggestions:

1. **Hyperparameter Tuning**: Use GridSearchCV or RandomizedSearchCV
2. **Feature Engineering**: Create interaction terms or polynomial features
3. **Ensemble Methods**: Combine multiple models with VotingClassifier
4. **Cross-validation**: Use k-fold CV to get more reliable estimates
5. **Feature Selection**: Remove irrelevant features
6. **Regularization**: Add L1/L2 regularization to prevent overfitting

Please provide more specific performance issues for targeted suggestions.
Issues can be: "low accuracy", "overfitting", "slow training", "unbalanced classes"
"""

        output = f"Optimization Suggestions for: {performance_issue}\n"
        output += "=" * 60 + "\n\n"
        output += suggestion_text
        output += "\n\nTo apply these optimizations:\n"
        output += "1. Copy the relevant code snippets\n"
        output += "2. Integrate them into your pipeline\n"
        output += "3. Re-run the pipeline with execute_ml_code\n"
        output += "4. Compare the results\n"

        return output

    except Exception as e:
        return f"Error generating optimization suggestions: {e!s}"
