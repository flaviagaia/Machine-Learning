from __future__ import annotations

from dataclasses import dataclass

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET_COLUMN = "defaulted"

NUMERIC_FEATURES = [
    "age",
    "annual_income",
    "employment_years",
    "credit_history_years",
    "loan_amount",
    "loan_term_months",
    "interest_rate",
    "monthly_debt",
    "debt_to_income",
    "credit_utilization",
    "late_payments_12m",
    "recent_credit_inquiries",
    "existing_loans",
    "savings_balance",
]

CATEGORICAL_FEATURES = [
    "home_ownership",
    "loan_purpose",
    "employment_type",
    "region",
]


@dataclass(frozen=True)
class CandidateModel:
    name: str
    pipeline: Pipeline


def build_preprocessor() -> ColumnTransformer:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )


def build_candidate_models(random_state: int = 42) -> list[CandidateModel]:
    preprocessor = build_preprocessor()

    logistic = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    solver="lbfgs",
                    random_state=random_state,
                ),
            ),
        ]
    )

    random_forest = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=250,
                    max_depth=8,
                    min_samples_leaf=8,
                    class_weight="balanced_subsample",
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    gradient_boosting = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                GradientBoostingClassifier(
                    n_estimators=180,
                    learning_rate=0.06,
                    max_depth=3,
                    random_state=random_state,
                ),
            ),
        ]
    )

    return [
        CandidateModel(name="logistic_regression", pipeline=logistic),
        CandidateModel(name="random_forest", pipeline=random_forest),
        CandidateModel(name="gradient_boosting", pipeline=gradient_boosting),
    ]
