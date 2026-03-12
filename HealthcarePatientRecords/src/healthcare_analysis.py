import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


def load_data(path):
    """Load dataset"""
    df = pd.read_csv(path)
    return df


def clean_data(df):
    """Handle missing values"""
    cols = ['Glucose','BloodPressure','SkinThickness','Insulin','BMI']
    df[cols] = df[cols].replace(0, np.nan)
    df.fillna(df.mean(), inplace=True)
    return df


def create_age_groups(df):
    """Create age group column"""
    df["AgeGroup"] = pd.cut(
        df["Age"],
        bins=[20,30,40,50,60,80],
        labels=["20-30","30-40","40-50","50-60","60+"]
    )
    return df


def mean_glucose_by_age(df):
    """Calculate mean glucose per age group"""
    mean_glucose = df.groupby("AgeGroup")["Glucose"].mean()
    return mean_glucose


def anova_test(df):
    """Perform ANOVA test on BMI"""
    bmi_diabetes = df[df["Outcome"] == 1]["BMI"]
    bmi_no_diabetes = df[df["Outcome"] == 0]["BMI"]

    result = stats.f_oneway(bmi_diabetes, bmi_no_diabetes)
    return result


def train_model(df):
    """Train regression model"""
    X = df.drop(["Outcome","AgeGroup"], axis=1)
    y = df["Outcome"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return y_test, y_pred, mse, r2


def visualize_mean_glucose(mean_glucose):
    """Bar chart: Mean glucose by age group"""
    colors = ["skyblue","orange","green","red","purple"]

    plt.figure(figsize=(8,5))
    mean_glucose.plot(kind="bar", color=colors)

    plt.title("Mean Glucose Levels by Age Group")
    plt.xlabel("Age Group")
    plt.ylabel("Mean Glucose Level")

    plt.tight_layout()
    plt.show()


def visualize_bmi_distribution(df):
    """BMI distribution by diabetes outcome"""
    plt.figure(figsize=(6,4))

    sns.boxplot(
        x="Outcome",
        y="BMI",
        data=df,
        palette=["skyblue","orange"]
    )

    plt.title("BMI Distribution by Diabetes Outcome")
    plt.show()


def visualize_prediction(y_test, y_pred):
    """Actual vs predicted scatter"""
    plt.figure(figsize=(6,4))

    plt.scatter(
        y_test + np.random.normal(0,0.02,len(y_test)),
        y_pred,
        color="purple",
        alpha=0.6
    )

    plt.xlabel("Actual Outcome")
    plt.ylabel("Predicted Outcome")
    plt.title("Actual vs Predicted Diabetes Outcome")

    plt.show()


def pairplot_visualization(df):
    """Pairplot of health indicators"""
    sns.pairplot(
        df[['Glucose','BMI','BloodPressure','Insulin','Age','Outcome']],
        hue='Outcome',
        palette=['skyblue','orange']
    )
    plt.show()


def glucose_distribution(df):
    """Glucose distribution histogram"""
    plt.figure(figsize=(6,4))

    sns.histplot(
        data=df,
        x="Glucose",
        hue="Outcome",
        bins=30,
        palette=["skyblue","orange"],
        kde=True
    )

    plt.title("Glucose Distribution by Diabetes Outcome")
    plt.show()


def bmi_vs_age(df):
    """Scatter: BMI vs Age"""
    plt.figure(figsize=(6,4))

    sns.scatterplot(
        data=df,
        x="Age",
        y="BMI",
        hue="Outcome",
        palette=["skyblue","orange"]
    )

    plt.title("BMI vs Age by Diabetes Outcome")
    plt.show()


def main():

    df = load_data("../data/diabetes.csv")

    df = clean_data(df)

    df = create_age_groups(df)

    mean_glucose = mean_glucose_by_age(df)

    print("Mean Glucose by Age Group:\n", mean_glucose)

    print("\nANOVA Result:", anova_test(df))

    y_test, y_pred, mse, r2 = train_model(df)

    print("\nModel Performance")
    print("MSE:", mse)
    print("R2 Score:", r2)

    visualize_mean_glucose(mean_glucose)

    visualize_bmi_distribution(df)

    visualize_prediction(y_test, y_pred)

    pairplot_visualization(df)

    glucose_distribution(df)

    bmi_vs_age(df)


if __name__ == "__main__":
    main()