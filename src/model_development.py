import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve
from sklearn.pipeline import Pipeline

# Create directories for saving results
results_dir = 'results'
figures_dir = os.path.join(results_dir, 'figures')
models_dir = os.path.join(results_dir, 'models')

os.makedirs(results_dir, exist_ok=True)
os.makedirs(figures_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)

# Load preprocessed data
# Assuming the preprocessed data is available as a CSV file
# Replace with your actual data loading code
def load_data(file_path='data/preprocessed_data.csv'):
    """Load preprocessed data from CSV file."""
    data = pd.read_csv(file_path)
    return data

def prepare_data(data):
    """Prepare data for modeling by splitting features and target."""
    # Assuming 'status_label' is the target column (1 for bankruptcy, 0 for non-bankruptcy)
    # and X1-X18 are the feature columns
    X = data.drop(['status_label', 'company_name', 'year'], axis=1, errors='ignore')
    y = data['status_label']
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    return X_train, X_test, y_train, y_test

def train_random_forest(X_train, y_train):
    """Train a Random Forest model with hyperparameter tuning."""
    # Create a pipeline with preprocessing and model
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(random_state=42))
    ])
    
    # Define hyperparameter grid
    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [None, 10, 20],
        'classifier__min_samples_split': [2, 5, 10],
        'classifier__min_samples_leaf': [1, 2, 4]
    }
    
    # Perform grid search with cross-validation
    grid_search = GridSearchCV(
        pipeline, param_grid, cv=5, scoring='roc_auc', n_jobs=-1, verbose=1
    )
    
    # Fit the model
    grid_search.fit(X_train, y_train)
    
    # Get the best model
    best_model = grid_search.best_estimator_
    
    # Print best parameters
    print("Best parameters:", grid_search.best_params_)
    print("Best cross-validation score:", grid_search.best_score_)
    
    return best_model

def evaluate_model(model, X_test, y_test):
    """Evaluate the model on test data."""
    # Make predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)
    
    # Calculate ROC AUC
    roc_auc = roc_auc_score(y_test, y_prob)
    print(f"\nROC AUC: {roc_auc:.4f}")
    
    # Plot ROC curve
    plt.figure(figsize=(10, 8))
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(figures_dir, 'roc_curve.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot Precision-Recall curve
    plt.figure(figsize=(10, 8))
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    plt.plot(recall, precision, label='Precision-Recall curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.savefig(os.path.join(figures_dir, 'precision_recall_curve.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # Return evaluation metrics
    return {
        'confusion_matrix': cm,
        'roc_auc': roc_auc,
        'fpr': fpr,
        'tpr': tpr,
        'precision': precision,
        'recall': recall
    }

def save_model(model, filename):
    """Save the trained model to disk."""
    import joblib
    joblib.dump(model, os.path.join(models_dir, filename))
    print(f"Model saved to {os.path.join(models_dir, filename)}")

def main():
    """Main function to run the model development pipeline."""
    print("Loading data...")
    data = load_data()
    
    print("Preparing data...")
    X_train, X_test, y_train, y_test = prepare_data(data)
    
    print("Training Random Forest model...")
    rf_model = train_random_forest(X_train, y_train)
    
    print("Evaluating model...")
    evaluation_metrics = evaluate_model(rf_model, X_test, y_test)
    
    print("Saving model...")
    save_model(rf_model, 'random_forest_model.joblib')
    
    # Save best feature set for explanation
    X_best = X_train
    
    # Save data for model explanation
    import joblib
    joblib.dump({
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'X_best': X_best,
        'model': rf_model
    }, os.path.join(models_dir, 'model_data.joblib'))
    
    print("Model development completed successfully!")

if __name__ == "__main__":
    main()
