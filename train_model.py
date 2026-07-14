import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib


# Dataset read
print("Loading dataset...")

df = pd.read_csv("creditcard.csv")

print("Dataset loaded successfully")


# Remove duplicates
df = df.drop_duplicates()


# Remove missing values
df = df.dropna()


# Convert text columns into numbers
le = LabelEncoder()

for col in df.select_dtypes(include='object').columns:
    df[col] = le.fit_transform(df[col])


# Features and Target

X = df.iloc[:, :-1]

y = df.iloc[:, -1]


# Split data

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Train model

print("Training model...")

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)


# Model accuracy

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy)


# Save model

joblib.dump(model, "model.pkl")


print("Model trained successfully")
print("Model saved as model.pkl")