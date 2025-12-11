"""
Helper script to save your trained models from the Jupyter notebook.

Instructions:
1. Run your entire notebook to train all models
2. Then run this script in a new cell at the end of your notebook

This will save the best performing model (Random Forest) and the scaler.
"""

import pickle
import os

def save_models():
    """Save the trained model and scaler for deployment."""
    
    try:
        # Get current directory
        current_dir = os.getcwd()
        print(f"Current directory: {current_dir}")
        
        # Save Random Forest model (best performer)
        if 'RF_model' in globals():
            with open('best_model.pkl', 'wb') as f:
                pickle.dump(RF_model, f)
            print("✅ Random Forest model saved as 'best_model.pkl'")
        else:
            print("❌ RF_model not found. Please train the Random Forest model first.")
            return False
        
        # Save StandardScaler
        if 'scaler' in globals():
            with open('scaler.pkl', 'wb') as f:
                pickle.dump(scaler, f)
            print("✅ StandardScaler saved as 'scaler.pkl'")
        else:
            print("❌ Scaler not found. Please ensure StandardScaler is defined.")
            return False
        
        # Verify files were created
        if os.path.exists('best_model.pkl') and os.path.exists('scaler.pkl'):
            model_size = os.path.getsize('best_model.pkl') / (1024 * 1024)  # MB
            scaler_size = os.path.getsize('scaler.pkl') / (1024 * 1024)  # MB
            
            print("\n📊 File Information:")
            print(f"   - best_model.pkl: {model_size:.2f} MB")
            print(f"   - scaler.pkl: {scaler_size:.2f} MB")
            print(f"   - Total size: {model_size + scaler_size:.2f} MB")
            
            if model_size > 100:
                print("\n⚠️  Warning: Model file is large (>100MB)")
                print("   Consider using Git LFS for GitHub deployment")
            
            print("\n✨ Models saved successfully!")
            print("   You can now deploy your app using these files.")
            return True
        else:
            print("❌ Error: Files were not created properly.")
            return False
            
    except Exception as e:
        print(f"❌ Error saving models: {str(e)}")
        return False

# Run the function
if __name__ == "__main__":
    save_models()
else:
    # When executed from notebook
    save_models()
