import argparse
import os
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def train():
    print("DEBUG: Starting training with fixed argparse!")
    # 1. Parse Arguments
    # SageMaker passes specific arguments for input/output directories
    parser = argparse.ArgumentParser()
    
    # SageMaker automatically sets these environment variables
    # We use defaults so it works locally on your laptop too
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR', 'models'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', 'data'))
    
    args, _ = parser.parse_known_args()

    # 2. Load Data
    
    file_path = os.path.join(args.train,"Airline.csv")
    
    
    df = pd.read_csv(file_path)

    # 3. Preprocessing
    # Drop rows with missing values
    df = df.dropna()
    
    # Drop irrelevant columns if they exist
    drop_cols = ['Unnamed: 0', 'id']
    # Pandas ignores columns that don't exist if you use errors='ignore'
    df = df.drop(columns=drop_cols, errors='ignore')

    # Encode Target Variable (satisfaction)
    # satisfied -> 1, neutral or dissatisfied -> 0
    le = LabelEncoder()
    df['satisfaction'] = le.fit_transform(df['satisfaction'])

    # One-Hot Encode Categorical Features (Gender, Type of Travel, etc.)
    df = pd.get_dummies(df)

    # Split Data
    X = df.drop('satisfaction', axis=1)
    y = df['satisfaction']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. Train Model
    
    model = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
    model.fit(X_train, y_train)

    # 5. Evaluate
    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    

    # 6. Save Model
    # Important: SageMaker reads the model from this specific path to deploy it later
    model_path = os.path.join(args.model_dir, "model.joblib")
    joblib.dump(model, model_path)
   
if __name__ == '__main__':
    train()