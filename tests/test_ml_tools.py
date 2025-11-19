"""Tests for ML tools."""

import pytest

from bioagents.tools.ml_tools import (
    design_ml_pipeline,
    optimize_ml_pipeline,
)


def test_design_ml_pipeline_basic():
    """Test basic ML pipeline design."""
    result = design_ml_pipeline.invoke(
        {
            "task_description": "Protein classification",
            "data_description": "CSV with protein features",
            "features": "length, gc_content, hydrophobicity",
            "target": "protein_type",
            "algorithm": "random_forest",
        }
    )

    assert isinstance(result, str)
    assert "import pandas" in result
    assert "import numpy" in result
    assert "RandomForestClassifier" in result
    assert "train_test_split" in result
    assert "protein_type" in result


def test_design_ml_pipeline_xgboost():
    """Test XGBoost pipeline design."""
    result = design_ml_pipeline.invoke(
        {
            "task_description": "Gene expression prediction",
            "data_description": "Gene expression data",
            "features": "gene1, gene2, gene3",
            "target": "expression_level",
            "algorithm": "xgboost",
        }
    )

    assert isinstance(result, str)
    assert "xgboost" in result.lower() or "XGBClassifier" in result or "GradientBoosting" in result


def test_design_ml_pipeline_logistic():
    """Test Logistic Regression pipeline design."""
    result = design_ml_pipeline.invoke(
        {
            "task_description": "Binary classification",
            "data_description": "Feature data",
            "features": "f1, f2, f3",
            "target": "label",
            "algorithm": "logistic_regression",
        }
    )

    assert isinstance(result, str)
    assert "LogisticRegression" in result


def test_optimize_ml_pipeline_low_accuracy():
    """Test optimization suggestions for low accuracy."""
    result = optimize_ml_pipeline.invoke(
        {
            "current_code": "model = RandomForestClassifier()",
            "performance_issue": "low accuracy",
            "optimization_strategy": "auto",
        }
    )

    assert isinstance(result, str)
    assert "accuracy" in result.lower()
    assert any(term in result for term in ["GridSearchCV", "hyperparameter", "feature"])


def test_optimize_ml_pipeline_overfitting():
    """Test optimization suggestions for overfitting."""
    result = optimize_ml_pipeline.invoke(
        {
            "current_code": "model = RandomForestClassifier()",
            "performance_issue": "overfitting",
            "optimization_strategy": "auto",
        }
    )

    assert isinstance(result, str)
    assert "overfitting" in result.lower()
    assert any(term in result for term in ["regularization", "max_depth", "cross"])


def test_optimize_ml_pipeline_slow_training():
    """Test optimization suggestions for slow training."""
    result = optimize_ml_pipeline.invoke(
        {
            "current_code": "model = RandomForestClassifier()",
            "performance_issue": "slow training",
            "optimization_strategy": "auto",
        }
    )

    assert isinstance(result, str)
    assert any(term in result for term in ["faster", "reduce", "feature selection"])


def test_optimize_ml_pipeline_imbalanced():
    """Test optimization suggestions for class imbalance."""
    result = optimize_ml_pipeline.invoke(
        {
            "current_code": "model = RandomForestClassifier()",
            "performance_issue": "unbalanced classes",
            "optimization_strategy": "auto",
        }
    )

    assert isinstance(result, str)
    assert any(term in result for term in ["class_weight", "SMOTE", "balance"])


def test_design_ml_pipeline_multiple_features():
    """Test pipeline design with multiple features."""
    result = design_ml_pipeline.invoke(
        {
            "task_description": "Complex classification",
            "data_description": "Multi-feature dataset",
            "features": "f1, f2, f3, f4, f5, f6, f7, f8, f9, f10",
            "target": "outcome",
            "algorithm": "auto",
        }
    )

    assert isinstance(result, str)
    assert "f1" in result or "['f1'" in result
    assert "outcome" in result


# Skip Docker-dependent tests if Docker is not available
def check_docker_available():
    """Check if Docker is available."""
    try:
        import docker

        client = docker.from_env()
        client.ping()
        return True
    except Exception:
        return False


@pytest.mark.skipif(not check_docker_available(), reason="Docker not available")
def test_execute_ml_code_simple():
    """Test simple code execution (requires Docker)."""
    from bioagents.tools.ml_tools import execute_ml_code

    code = """
print("Hello from ML sandbox!")
import numpy as np
print(f"NumPy version: {np.__version__}")
"""

    result = execute_ml_code.invoke(
        {
            "code": code,
            "data_csv": None,
            "additional_files": None,
        }
    )

    assert isinstance(result, str)
    assert "Hello from ML sandbox!" in result or "completed" in result.lower()


@pytest.mark.skipif(not check_docker_available(), reason="Docker not available")
def test_execute_ml_code_with_data():
    """Test code execution with CSV data (requires Docker)."""
    from bioagents.tools.ml_tools import execute_ml_code

    code = """
import pandas as pd
df = pd.read_csv('data.csv')
print(f"Data shape: {df.shape}")
print(df.head())
"""

    data_csv = "x,y\n1,2\n3,4\n5,6\n"

    result = execute_ml_code.invoke(
        {
            "code": code,
            "data_csv": data_csv,
            "additional_files": None,
        }
    )

    assert isinstance(result, str)
    assert "completed" in result.lower() or "Data shape" in result
