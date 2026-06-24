import os
import time
import joblib
import shap
import numpy as np

def main():
    print("--- STEP 1: Loading Model Artifacts ---")
    artifacts_path = 'model_artifacts.joblib'
    if not os.path.exists(artifacts_path):
        raise FileNotFoundError(f"Required model artifacts file '{artifacts_path}' not found in the workspace.")
        
    data = joblib.load(artifacts_path)
    model = data['model']
    scaler = data['scaler']
    X_test = data['X_test']
    feature_names = data['feature_names']
    
    print(f"Loaded model: {type(model)}")
    print(f"Test set shape: {X_test.shape}")
    
    print("\n--- STEP 2: Standardizing Test Set ---")
    X_test_scaled = scaler.transform(X_test)
    
    print("\n--- STEP 3: Initializing TreeExplainer ---")
    t0 = time.time()
    explainer = shap.TreeExplainer(model)
    print(f"Explainer created in {time.time() - t0:.3f} seconds")
    
    print("\n--- STEP 4: Calculating SHAP Values for Test Set (2000 instances) ---")
    t1 = time.time()
    explanation = explainer(X_test_scaled)
    print(f"Computed SHAP values in {time.time() - t1:.3f} seconds")
    
    # Verify shape
    # explanation.values should have shape (2000, 11, 2)
    print(f"Explanation values shape: {explanation.values.shape}")
    
    # Class 1 corresponds to 'Exited' (churn)
    shap_values_class1 = explanation.values[:, :, 1]
    base_value_class1 = explainer.expected_value[1]
    
    print(f"Class 1 SHAP values shape: {shap_values_class1.shape}")
    print(f"Class 1 base value (expected value): {base_value_class1:.6f}")
    
    print("\n--- STEP 5: Saving SHAP Artifacts to Disk ---")
    output_path = 'shap_values.joblib'
    joblib.dump({
        'shap_values_class1': shap_values_class1,
        'base_value_class1': base_value_class1
    }, output_path)
    
    print(f"Successfully saved SHAP artifacts to '{output_path}'")

if __name__ == '__main__':
    main()
