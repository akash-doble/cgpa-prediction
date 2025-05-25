import pandas as pd
import argparse
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess(input_path, output_dir):
    # Load the input CSV
    df = pd.read_csv(input_path)

    # Extract features and target
    X = df[['hours_study', 'attendance']]
    y = df[['cgpa']]

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Combine scaled features with target
    df_scaled = pd.DataFrame(X_scaled, columns=['hours_study', 'attendance'])
    df_scaled['cgpa'] = y.values

    # Split into train/test sets
    train, test = train_test_split(df_scaled, test_size=0.2, random_state=42)

    # Save the splits
    os.makedirs(output_dir, exist_ok=True)
    train.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
    test.to_csv(os.path.join(output_dir, 'test.csv'), index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-data", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="/opt/ml/processing/output")
    args = parser.parse_args()

    preprocess(args.input_data, args.output_dir)
