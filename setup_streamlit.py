"""
Helper script to prepare functions for Streamlit app.
Run this in your notebook or as a script after loading your data.

This will save the necessary functions and data for the Streamlit app.
"""

import pickle
import pandas as pd

def save_for_streamlit():
    """Save functions and data for Streamlit app"""
    try:
        # Save the prediction dataframe
        if 'full_pivot_df' in globals():
            with open('full_pivot_df.pkl', 'wb') as f:
                pickle.dump(full_pivot_df, f)
            print("✓ Saved full_pivot_df.pkl")
        elif 'test_pivot_df' in globals():
            with open('test_pivot_df.pkl', 'wb') as f:
                pickle.dump(test_pivot_df, f)
            print("✓ Saved test_pivot_df.pkl")
        
        # Save functions (we'll need to import them in streamlit)
        print("✓ Functions will be imported directly from notebook context")
        print("\nTo use Streamlit app:")
        print("1. Make sure this script has run")
        print("2. Run: streamlit run streamlit_app.py")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    save_for_streamlit()

