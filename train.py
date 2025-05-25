import argparse
import pandas as pd
import joblib
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def train(train_path, model_dir):
    # Load the training data
    df = pd.read_csv(train_path)

    # Split into features and target
    X = df[['hours_study', 'attendance']]
    y = df['cgpa']

    # Train the model
    model = LinearRegression()
    model.fit(X, y)

    # Evaluate on training data
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    print(f"Training MSE: {mse}")

    # Save the model
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model, os.path.join(model_dir, "model.joblib"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str, default='/opt/ml/input/data/train/train.csv')
    parser.add_argument('--model-dir', type=str, default='/opt/ml/model')
    args = parser.parse_args()

    train(args.train, args.model_dir)
