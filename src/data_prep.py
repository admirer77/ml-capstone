import pandas as pd
import seaborn as sns

def load_titanic():
    df = sns.load_dataset("titanic")
    return df

def clean_titanic(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["age"] = df["age"].fillna(df["age"].median())
    df["embarked"] = df["embarked"].fillna(df["embarked"].mode()[0])
    df = df.drop(columns=["deck", "embark_town", "alive", "who", "adult_male", "class"])
    df["sex"] = df["sex"].map({"male": 0, "female": 1})
    df = pd.get_dummies(df, columns=["embarked"], drop_first=True)
    df = df.dropna()
    return df