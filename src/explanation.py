import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import shap
from sklearn.ensemble import RandomForestClassifier

# Create directories for saving results
results_dir = 'results'
figures_dir = os.path.join(results_dir, 'figures')
os.makedirs(figures_dir, exist_ok=True)

def load_model_data():
    """Load the model and data for explanation."""
    model_data = joblib.load(os.path.join('results', 'models', 'model_data.joblib'))
    return model_data

def perform_shap_analysis(model, X_best, X_test, y_test):
    """Perform SHAP analysis for model interpretation."""
    print("\n=== SHAP Analysis for Best Model ===")
    
    # Identify the best model
    best_model = model
    best_model_name = "Random Forest"
    
    # Create a Background Sample for SHAP
    X_background = X_best.sample(min(100, len(X_best)), random_state=42)
    
    # Create the SHAP Explainer
    if isinstance(best_model, RandomForestClassifier) or hasattr(best_model, 'estimators_'):
        # For tree-based models, use TreeExplainer
        explainer = shap.TreeExplainer(best_model)
        
        # Sample test data for analysis
        X_test_sample = X_test.sample(min(200, len(X_test)), random_state=42)
        
        # Calculate SHAP values
        shap_values = explainer.shap_values(X_test_sample)
        
        # For classification models, shap_values is a list with one element per class
        if isinstance(shap_values, list):
            # Use values for the positive class (bankruptcy)
            shap_values_to_plot = shap_values[1]
            expected_value = explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value
        else:
            shap_values_to_plot = shap_values
            expected_value = explainer.expected_value
        
        # Print diagnostic information
        print(f"Shape of shap_values_to_plot: {np.array(shap_values_to_plot).shape}")
        print(f"Shape of X_test_sample: {X_test_sample.shape}")
        
        # Handle 3D SHAP values by taking the mean across the last dimension if needed
        if len(np.array(shap_values_to_plot).shape) == 3:
            print("Detected 3D SHAP values, taking mean across last dimension")
            shap_values_to_plot = np.mean(shap_values_to_plot, axis=2)
            print(f"New shape of shap_values_to_plot: {shap_values_to_plot.shape}")
        
        # Check for very small SHAP values and normalize if needed
        shap_max = np.max(np.abs(shap_values_to_plot))
        print(f"Maximum absolute SHAP value: {shap_max}")
        
        # If SHAP values are very small, normalize them
        if shap_max < 1e-5:
            print("SHAP values are very small. Normalizing for better visualization...")
            # Normalize SHAP values to a more reasonable range
            shap_values_to_plot = shap_values_to_plot / shap_max * 1.0
            print(f"Normalized SHAP values. New maximum: {np.max(np.abs(shap_values_to_plot))}")
        
        # --- Global Explanation ---
        # 1. Feature Importance Bar Plot
        plt.figure(figsize=(12, 10))
        
        # Convert feature names to numpy array to avoid indexing issues
        feature_names = np.array(X_test_sample.columns.tolist())
        
        # Use the newer SHAP API with Explanation objects
        try:
            # Create a SHAP Explanation object
            explanation = shap.Explanation(
                values=shap_values_to_plot,
                base_values=np.ones(len(X_test_sample)) * expected_value,
                data=X_test_sample.values,
                feature_names=feature_names
            )
            
            # Use the newer plots API
            shap.plots.bar(explanation, show=False)
            plt.title(f"Feature Importance Based on SHAP Values - {best_model_name}", fontsize=14)
            plt.tight_layout()
            plt.savefig(os.path.join(figures_dir, f'{best_model_name.replace(" ", "_")}_shap_importance.png'), dpi=300, bbox_inches='tight')
            plt.show()
            
            # 2. Summary Dot Plot
            plt.figure(figsize=(14, 12))
            shap.plots.beeswarm(explanation, show=False)
            plt.title(f"SHAP Summary Plot - {best_model_name}", fontsize=14)
            plt.tight_layout()
            plt.savefig(os.path.join(figures_dir, f'{best_model_name.replace(" ", "_")}_shap_summary.png'), dpi=300, bbox_inches='tight')
            plt.show()
            
        except Exception as e:
            print(f"Error using newer SHAP API: {e}")
            print("Falling back to legacy plotting functions")
            
            # Fall back to legacy functions
            plt.figure(figsize=(12, 10))
            shap.summary_plot(
                shap_values_to_plot,
                X_test_sample,
                feature_names=feature_names,
                plot_type="bar",
                show=False
            )
            plt.title(f"Feature Importance Based on SHAP Values - {best_model_name}", fontsize=14)
            plt.tight_layout()
            plt.savefig(os.path.join(figures_dir, f'{best_model_name.replace(" ", "_")}_shap_importance.png'), dpi=300, bbox_inches='tight')
            plt.show()
            
            # 2. Summary Dot Plot
            plt.figure(figsize=(14, 12))
            shap.summary_plot(
                shap_values_to_plot,
                X_test_sample,
                feature_names=feature_names,
                show=False
            )
            plt.title(f"SHAP Summary Plot - {best_model_name}", fontsize=14)
            plt.tight_layout()
            plt.savefig(os.path.join(figures_dir, f'{best_model_name.replace(" ", "_")}_shap_summary.png'), dpi=300, bbox_inches='tight')
            plt.show()
        
        # 3. Calculate feature importance for dependence plots
        # Ensure feature_importance is 1-dimensional
        feature_importance = np.abs(shap_values_to_plot).mean(axis=0)
        print(f"Shape of feature_importance: {feature_importance.shape}")
        print(f"Number of columns in X_test_sample: {len(X_test_sample.columns)}")
        
        if len(feature_importance.shape) > 1:
            feature_importance = feature_importance.flatten()
        
        # Check if lengths match
        if len(feature_importance) != len(X_test_sample.columns):
            print("WARNING: Feature importance length doesn't match number of columns!")
            # Truncate to the shorter length
            min_length = min(len(feature_importance), len(X_test_sample.columns))
            feature_importance = feature_importance[:min_length]
            feature_names_list = X_test_sample.columns[:min_length].tolist()
        else:
            feature_names_list = X_test_sample.columns.tolist()
        
        # Create DataFrame of feature importance
        feature_importance_df = pd.DataFrame({
            'Feature': feature_names_list,
            'Importance': feature_importance
        }).sort_values('Importance', ascending=False)
        
        # Print top features and their importance values
        print("\nTop 10 features by importance:")
        print(feature_importance_df.head(10))
        
        # 4. Dependence Plots for top features - using manual scatter plots
        top_features = feature_importance_df.head(3)['Feature'].values
        
        for feature in top_features:
            try:
                # Find the index of the feature
                feature_idx = list(feature_names_list).index(feature)
                print(f"Creating dependence plot for feature: {feature} (index: {feature_idx})")
                
                # Create a manual scatter plot
                plt.figure(figsize=(10, 8))
                plt.scatter(
                    X_test_sample[feature].values,
                    shap_values_to_plot[:, feature_idx],
                    alpha=0.6
                )
                plt.xlabel(feature)
                plt.ylabel(f"SHAP value for {feature}")
                plt.title(f"SHAP Dependence Plot: {feature} - {best_model_name}", fontsize=14)
                plt.tight_layout()
                plt.savefig(os.path.join(figures_dir, f'{best_model_name.replace(" ", "_")}_{feature}_dependence.png'), dpi=300, bbox_inches='tight')
                plt.show()
            except Exception as e:
                print(f"Error creating dependence plot for {feature}: {e}")
                continue
        
        # --- Local Explanation ---
        # Find a bankruptcy case for detailed analysis
        if np.sum(y_test == 1) > 0:
            # Find a bankruptcy case
            bankruptcy_indices = np.where(y_test == 1)[0]
            example_idx = bankruptcy_indices[0]  # Take the first bankruptcy case
            
            # Get the prediction and actual value
            example_X = X_test.iloc[example_idx:example_idx+1]
            example_y = y_test.iloc[example_idx]
            example_pred = best_model.predict(example_X)[0]
            
            print(f"\nLocal Explanation for a Bankruptcy Case (index {example_idx}):")
            print(f"Actual value: {example_y} (1 = Bankruptcy)")
            print(f"Predicted value: {example_pred}")
            
            # Calculate SHAP values for this instance
            if isinstance(best_model, RandomForestClassifier) or hasattr(best_model, 'estimators_'):
                example_shap_values = explainer.shap_values(example_X)
                
                if isinstance(example_shap_values, list):
                    example_shap_values = example_shap_values[1]  # Values for positive class
                
                # Handle 3D SHAP values for the example
                if len(np.array(example_shap_values).shape) == 3:
                    example_shap_values = np.mean(example_shap_values, axis=2)
                
                # Normalize example SHAP values if they're very small
                if shap_max < 1e-5:
                    example_shap_values = example_shap_values / shap_max * 1.0
                
                # 5. Print the top contributing features for this instance
                feature_contribution = pd.DataFrame({
                    'Feature': example_X.columns.tolist(),
                    'SHAP Value': example_shap_values[0],
                    'Feature Value': example_X.values[0]
                }).sort_values('SHAP Value', key=abs, ascending=False)
                
                print("\nTop Contributing Features for this Bankruptcy Case:")
                print(feature_contribution.head(10))
                
                # Save this information
                feature_contribution.to_csv(os.path.join(figures_dir, f'{best_model_name.replace(" ", "_")}_top_features_case_{example_idx}.csv'), index=False)
                
                # 6. Create a simple bar chart of feature contributions
                plt.figure(figsize=(12, 8))
                top_features = feature_contribution.head(10)
                plt.barh(top_features['Feature'], top_features['SHAP Value'])
                plt.xlabel('SHAP Value (Impact on Prediction)')
                plt.ylabel('Feature')
                plt.title(f"Top Features Influencing Bankruptcy Prediction - Case {example_idx}", fontsize=14)
                plt.tight_layout()
                plt.savefig(os.path.join(figures_dir, f'{best_model_name.replace(" ", "_")}_top_features_barplot.png'), dpi=300, bbox_inches='tight')
                plt.show()
                
                # 7. Force Plot - try both new and legacy approaches
                try:
                    # Try the newer API first with explicit handling of expected_value
                    plt.figure(figsize=(20, 3))
                    
                    # Make sure expected_value is a scalar
                    if isinstance(expected_value, (list, np.ndarray)):
                        base_value = float(expected_value[0]) if len(expected_value) > 0 else 0.0
                    else:
                        base_value = float(expected_value)
                    
                    print(f"Base value for force plot: {base_value}")
                    
                    # Create force plot with newer API
                    shap.plots.force(
                        base_value=base_value,
                        shap_values=example_shap_values[0],
                        features=example_X.iloc[0],
                        feature_names=example_X.columns.tolist(),
                        matplotlib=True,
                        show=False
                    )
                    plt.title(f"SHAP Force Plot: Factors Influencing Bankruptcy Prediction", fontsize=14)
                    plt.tight_layout()
                    plt.savefig(os.path.join(figures_dir, f'{best_model_name.replace(" ", "_")}_force_plot.png'), dpi=300, bbox_inches='tight')
                    plt.show()
                    
                except Exception as e:
                    print(f"Error with newer force plot API: {e}")
                    print("Trying alternative force plot approach...")
                    
                    try:
                        # Create a manual force-like plot
                        plt.figure(figsize=(20, 3))
                        
                        # Sort features by SHAP value
                        sorted_idx = np.argsort(example_shap_values[0])
                        sorted_values = example_shap_values[0][sorted_idx]
                        sorted_names = np.array(example_X.columns)[sorted_idx]
                        
                        # Create a horizontal bar chart
                        colors = ['red' if x < 0 else 'blue' for x in sorted_values]
                        plt.barh(range(len(sorted_values)), sorted_values, color=colors)
                        plt.yticks(range(len(sorted_values)), sorted_names)
                        plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
                        plt.xlabel('SHAP Value (negative = decreases prediction, positive = increases prediction)')
                        plt.title(f"SHAP Force Plot: Factors Influencing Bankruptcy Prediction", fontsize=14)
                        plt.tight_layout()
                        plt.savefig(os.path.join(figures_dir, f'{best_model_name.replace(" ", "_")}_force_plot.png'), dpi=300, bbox_inches='tight')
                        plt.show()
                        
                    except Exception as e2:
                        print(f"Error with alternative force plot: {e2}")
                        print("Skipping force plot.")
                
                # 8. Waterfall plot - try both new and legacy approaches
                try:
                    # Try the newer API first with explicit handling of expected_value
                    plt.figure(figsize=(12, 8))
                    
                    # Make sure expected_value is a scalar
                    if isinstance(expected_value, (list, np.ndarray)):
                        base_value = float(expected_value[0]) if len(expected_value) > 0 else 0.0
                    else:
                        base_value = float(expected_value)
                    
                    print(f"Base value for waterfall plot: {base_value}")
                    
                    # Get top 10 features by impact
                    top_indices = np.argsort(np.abs(example_shap_values[0]))[-10:]
                    
                    # Create a waterfall plot with the newer API
                    shap.plots.waterfall(
                        shap_values=example_shap_values[0][top_indices],
                        base_value=base_value,
                        feature_names=np.array(example_X.columns)[top_indices].tolist(),
                        show=False
                    )
                    plt.title(f"SHAP Waterfall Plot: Bankruptcy Case Analysis", fontsize=14)
                    plt.tight_layout()
                    plt.savefig(os.path.join(figures_dir, f'{best_model_name.replace(" ", "_")}_waterfall_plot.png'), dpi=300, bbox_inches='tight')
                    plt.show()
                    
                except Exception as e:
                    print(f"Error with newer waterfall plot API: {e}")
                    print("Trying legacy waterfall plot...")
                    
                    try:
                        # Fall back to legacy waterfall plot
                        plt.figure(figsize=(12, 8))
                        
                        # Sort features by absolute SHAP value
                        sorted_idx = np.argsort(np.abs(example_shap_values[0]))[::-1]
                        sorted_values = example_shap_values[0][sorted_idx]
                        sorted_names = np.array(example_X.columns)[sorted_idx]
                        
                        # Limit to top 10 features
                        sorted_values = sorted_values[:10]
                        sorted_names = sorted_names[:10]
                        
                        # Create a manual waterfall plot
                        plt.barh(range(len(sorted_values)), sorted_values)
                        plt.yticks(range(len(sorted_values)), sorted_names)
                        plt.xlabel('SHAP Value')
                        plt.title(f"SHAP Waterfall Plot: Bankruptcy Case Analysis", fontsize=14)
                        plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
                        plt.tight_layout()
                        plt.savefig(os.path.join(figures_dir, f'{best_model_name.replace(" ", "_")}_waterfall_plot.png'), dpi=300, bbox_inches='tight')
                        plt.show()
                        
                    except Exception as e2:
                        print(f"Error with legacy waterfall plot: {e2}")
                        print("Skipping waterfall plot.")
    
    else:
        # For non-tree models, use KernelExplainer
        print("Using KernelExplainer for non-tree model")
        
        # Create a summary dataset for the explainer
        X_summary = shap.kmeans(X_best, 50)  # Use 50 representative samples
        
        # Initialize the KernelExplainer
        explainer = shap.KernelExplainer(best_model.predict_proba, X_summary)
        
        # Sample test data for analysis (use fewer samples as KernelExplainer is slower)
        X_test_sample = X_test.sample(min(50, len(X_test)), random_state=42)
        
        # Calculate SHAP values (this may take some time)
        print("Calculating SHAP values with KernelExplainer (this may take a while)...")
        shap_values = explainer.shap_values(X_test_sample)
        
        # For classification, use values for the positive class
        if isinstance(shap_values, list):
            shap_values_to_plot = shap_values[1]  # Values for positive class
            expected_value = explainer.expected_value[1]
        else:
            shap_values_to_plot = shap_values
            expected_value = explainer.expected_value
        
        # Check for very small SHAP values and normalize if needed
        shap_max = np.max(np.abs(shap_values_to_plot))
        print(f"Maximum absolute SHAP value: {shap_max}")
        
        # If SHAP values are very small, normalize them
        if shap_max < 1e-5:
            print("SHAP values are very small. Normalizing for better visualization...")
            # Normalize SHAP values to a more reasonable range
            shap_values_to_plot = shap_values_to_plot / shap_max * 1.0
            print(f"Normalized SHAP values. New maximum: {np.max(np.abs(shap_values_to_plot))}")
    
    print("\nSHAP analysis completed and visualizations saved to results/figures directory.")
    
    return feature_importance_df

def analyze_feature_distributions(X_test, y_test, feature_importance_df, best_model_name="Random Forest"):
    """Analyze the distribution of top features by class."""
    print("\n=== Feature Value Distribution for Top Features ===")
    
    # Get top 5 features from SHAP analysis
    top_features = feature_importance_df.head(5)['Feature'].values
    
    # Create distribution plots for these features
    plt.figure(figsize=(15, 10))
    for i, feature in enumerate(top_features):
        plt.subplot(2, 3, i+1)
        
        # Plot distributions separately for each class
        for target_class in [0, 1]:
            class_data = X_test[y_test == target_class][feature]
            plt.hist(class_data, alpha=0.5, bins=20, 
                     label=f"Class {target_class}" + (" (Bankruptcy)" if target_class == 1 else " (Non-Bankruptcy)"))
        
        plt.xlabel(feature)
        plt.ylabel('Frequency')
        plt.title(f"Distribution of {feature}")
        plt.legend()
        plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, f'{best_model_name.replace(" ", "_")}_top_features_distribution.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary statistics for top features by class
    print("\nSummary statistics for top features by class:")
    for feature in top_features:
        print(f"\nFeature: {feature}")
        for target_class in [0, 1]:
            class_data = X_test[y_test == target_class][feature]
            print(f"Class {target_class}" + (" (Bankruptcy)" if target_class == 1 else " (Non-Bankruptcy)"))
            print(f"  Mean: {class_data.mean():.4f}")
            print(f"  Median: {class_data.median():.4f}")
            print(f"  Std Dev: {class_data.std():.4f}")
            print(f"  Min: {class_data.min():.4f}")
            print(f"  Max: {class_data.max():.4f}")

def main():
    """Main function to run the model explanation pipeline."""
    print("Loading model and data...")
    model_data = load_model_data()
    
    # Extract model and data
    model = model_data['model']
    X_best = model_data['X_best']
    X_test = model_data['X_test']
    y_test = model_data['y_test']
    
    print("Performing SHAP analysis...")
    feature_importance_df = perform_shap_analysis(model, X_best, X_test, y_test)
    
    print("Analyzing feature distributions...")
    analyze_feature_distributions(X_test, y_test, feature_importance_df)
    
    print("Model explanation completed successfully!")

if __name__ == "__main__":
    main()


