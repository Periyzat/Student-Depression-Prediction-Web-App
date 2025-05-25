import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
import shap

matplotlib.use("Agg")

df = pd.read_csv("dataset/Depression Student Dataset.csv")

cols = [
    "Have you ever had suicidal thoughts ?",
    "Family History of Mental Illness",
    "Depression",
]

df[cols] = (df[cols] == "Yes").astype(int)

df["Gender"] = (df["Gender"] == "Female").astype(int)
df["Sleep Duration"] = df["Sleep Duration"].apply(
    lambda x: (
        0
        if x == "Less than 5 hours"
        else 1 if x == "5-6 hours" else 2 if x == "7-8 hours" else 3
    )
)
df["Dietary Habits"] = df["Dietary Habits"].apply(
    lambda x: 0 if x == "Unhealthy" else 1 if x == "Moderate" else 2
)

X = df.drop(columns="Depression")
y = df["Depression"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train = np.array(X_train)
X_test = np.array(X_test)

model = tf.keras.models.load_model("model/my_model.keras")


# Function to create and save the plot
def create_plot(model, model_input):

    project_root = os.path.abspath(os.path.dirname(__file__))

    static_folder = os.path.join(project_root, "../static")
    os.makedirs(static_folder, exist_ok=True)
    plot_path = os.path.join(static_folder, "depression_plot.png")

    prediction = model.predict(model_input) 
    
    feature_names = [
        "Gender",
        "Age",
        "Academic Pressure",
        "Study Satisfaction",
        "Sleep Duration",
        "Dietary Habits",
        "Suicidal Thoughts",
        "Study Hours",
        "Financial Stress",
        "Mental Illness",
    ]

    model_input_df = pd.DataFrame(model_input, columns=feature_names)

    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(model_input_df)
    shap_values_instance = shap_values[0].values

    positive_contributions = {
        feature: value * 100
        for feature, value in zip(feature_names, shap_values_instance)
        if value >= 0
    }

    sorted_contributions = dict(
        sorted(positive_contributions.items(), key=lambda item: item[1], reverse=True)
    )

    contribution_df = pd.DataFrame(
        list(sorted_contributions.items()), columns=["Indicators", "%"]
    )

    plt.figure(figsize=(10, 6))
    sns.barplot(
        y="Indicators",
        x="%",
        data=contribution_df,
        palette="coolwarm",
    )
    plt.xlabel("Percentage (%))")
    plt.title("Main reasons of your depression")
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

    return prediction

