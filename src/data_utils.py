"""Data loading and preprocessing utilities for the Titanic survival model."""

import logging
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)


def load_titanic() -> pd.DataFrame:
    """Load the raw Titanic dataset."""
    logger.info("Loading Titanic dataset...")
    df = sns.load_dataset("titanic")
    logger.info("Loaded %d rows, %d columns", df.shape[0], df.shape[1])
    return df


def clean_titanic(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and encode the Titanic dataset for modeling."""
    logger.info("Cleaning dataset...")
    df = df.copy()
    df["age"] = df["age"].fillna(df["age"].median())
    df["embarked"] = df["embarked"].fillna(df["embarked"].mode()[0])
    df = df.drop(columns=["deck", "embark_town", "alive", "who", "adult_male", "class"])
    df["sex"] = df["sex"].map({"male": 0, "female": 1})
    df = pd.get_dummies(df, columns=["embarked"], drop_first=True)
    df = df.dropna()
    logger.info("Cleaned dataset shape: %s", df.shape)
    return df