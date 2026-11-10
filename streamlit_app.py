import streamlit as st
import pandas as pd
from datetime import datetime
import pickle
import os

# Page configuration
st.set_page_config(
    page_title="IoT Anomaly Analysis",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 IoT Security Anomaly Analysis")
st.markdown("Analyze security anomalies detected in IoT sensor data using LLM-powered explanations")

# Initialize session state
if 'prediction_df' not in st.session_state:
    st.session_state['prediction_df'] = None
if 'functions_loaded' not in st.session_state:
    st.session_state['functions_loaded'] = False

# Load data and functions
@st.cache_data
def load_prediction_data():
    """Try to load prediction dataframe from pickle or globals"""
    # Try pickle file first
    if os.path.exists('full_pivot_df.pkl'):
        with open('full_pivot_df.pkl', 'rb') as f:
            return pickle.load(f)
    elif os.path.exists('test_pivot_df.pkl'):
        with open('test_pivot_df.pkl', 'rb') as f:
            return pickle.load(f)
    return None

# Try to load functions
def load_functions():
    """Load required functions"""
    try:
        # Try importing from code_new module
        import code_new
        return (
            getattr(code_new, 'get_anomalies_by_date', None),
            getattr(code_new, 'explain_single_anomaly', None)
        )
    except:
        try:
            # Try from globals (if running from notebook)
            import __main__
            return (
                getattr(__main__, 'get_anomalies_by_date', None),
                getattr(__main__, 'explain_single_anomaly', None)
            )
        except:
            return None, None

# Load data
prediction_df = load_prediction_data()
if prediction_df is None:
    # Try to get from globals
    try:
        import __main__
        if hasattr(__main__, 'full_pivot_df'):
            prediction_df = __main__.full_pivot_df
        elif hasattr(__main__, 'test_pivot_df'):
            prediction_df = __main__.test_pivot_df
    except:
        pass

# Load functions from the module
try:
    from anomaly_functions import get_anomalies_by_date, explain_single_anomaly
    st.session_state['functions_loaded'] = True
except ImportError:
    # Try alternative loading methods
    get_anomalies_by_date, explain_single_anomaly = load_functions()
    if get_anomalies_by_date is None or explain_single_anomaly is None:
        st.error("❌ Required functions not found!")
        st.info("""
        **To fix this:**
        1. Make sure `anomaly_functions.py` exists in the same directory
        2. Or run the notebook cells that define these functions
        3. The functions should be: `get_anomalies_by_date` and `explain_single_anomaly`
        """)
        st.stop()

# Check if we have what we need
if prediction_df is None:
    st.error("❌ Prediction DataFrame not found!")
    st.info("""
    **To fix this:**
    1. Run your notebook to create `full_pivot_df` or `test_pivot_df`
    2. Save the dataframe to a pickle file:
       ```python
       import pickle
       with open('full_pivot_df.pkl', 'wb') as f:
           pickle.dump(full_pivot_df, f)
       ```
    3. Make sure the pickle file is in the same directory as this app
    """)
    st.stop()

# Sidebar for date input
st.sidebar.header("📅 Select Date")
date_input = st.sidebar.date_input(
    "Choose a date to analyze",
    value=datetime(2011, 6, 1),
    min_value=datetime(2011, 1, 1),
    max_value=datetime(2011, 12, 31)
)

# Convert date to string format
date_str = date_input.strftime("%Y-%m-%d")

# Main content
st.sidebar.markdown("---")
if st.sidebar.button("🔎 Find Anomalies", type="primary"):
    st.session_state['selected_date'] = date_str
    st.session_state['anomalies'] = None
    st.session_state['selected_anomaly'] = None
    st.session_state['selected_index'] = None

# Get anomalies for the selected date
if 'selected_date' in st.session_state and st.session_state['selected_date']:
    date_str = st.session_state['selected_date']
    
    try:
        anomalies = get_anomalies_by_date(date_str, prediction_df)
        
        if anomalies is None or len(anomalies) == 0:
            st.warning(f"⚠️ No anomalies detected on {date_str}")
            st.info("Try selecting a different date.")
        else:
            st.session_state['anomalies'] = anomalies
            
            st.success(f"✅ Found {len(anomalies)} anomaly/anomalies on {date_str}")
            
            # Display anomalies in a table
            st.subheader("📋 Detected Anomalies")
            
            # Create a summary table
            anomaly_summary = []
            for idx, (_, anomaly) in enumerate(anomalies.iterrows()):
                timestamp = anomaly['Timestamp']
                activity = anomaly.get('Activity', 'Unknown')
                confidence = anomaly.get('Prediction_Confidence', 0)
                
                anomaly_summary.append({
                    'Index': idx,
                    'Time': pd.to_datetime(timestamp).strftime('%Y-%m-%d %H:%M:%S') if pd.notna(timestamp) else 'N/A',
                    'Activity': activity,
                    'Confidence': f"{confidence:.2%}"
                })
            
            summary_df = pd.DataFrame(anomaly_summary)
            st.dataframe(summary_df, use_container_width=True, hide_index=True)
            
            # Anomaly selection
            st.subheader("🎯 Select Anomaly to Analyze")
            anomaly_indices = list(range(len(anomalies)))
            selected_index = st.selectbox(
                "Choose an anomaly to get detailed LLM explanation:",
                options=anomaly_indices,
                format_func=lambda x: f"Anomaly #{x+1} - {anomaly_summary[x]['Time']} ({anomaly_summary[x]['Activity']})"
            )
            
            st.session_state['selected_index'] = selected_index
            
            if st.button("🚀 Generate LLM Explanation", type="primary"):
                with st.spinner("🤖 Generating explanation with TinyLlama... This may take a minute."):
                    try:
                        # Try to get SHAP-related variables if available
                        shap_vals = None
                        X_df = None
                        shap_explainer = None
                        shap_scaler = None
                        
                        try:
                            import __main__
                            shap_vals = getattr(__main__, 'shap_values', None)
                            X_df = getattr(__main__, 'X', None)
                            shap_explainer = getattr(__main__, 'explainer', None)
                            shap_scaler = getattr(__main__, 'scaler', None)
                        except:
                            pass
                        
                        result = explain_single_anomaly(
                            date_str, 
                            selected_index, 
                            prediction_df,
                            shap_values=shap_vals,
                            X=X_df,
                            explainer=shap_explainer,
                            scaler=shap_scaler
                        )
                        
                        if result and 'explanation' in result and result['explanation']:
                            st.session_state['selected_anomaly'] = result
                            
                            # Display the explanation
                            st.subheader("📊 Anomaly Analysis Report")
                            st.markdown("---")
                            
                            # Show explanation with proper formatting
                            explanation = result['explanation']
                            
                            # Display in a nice text area or markdown
                            st.text_area(
                                "Full Report",
                                explanation,
                                height=400,
                                disabled=True,
                                label_visibility="collapsed"
                            )
                            
                            # Also show in expandable sections
                            with st.expander("📄 View Formatted Report", expanded=True):
                                # Split explanation into sections
                                lines = explanation.split('\n')
                                for line in lines:
                                    if line.strip():
                                        if line.startswith('ANOMALY ANALYSIS REPORT'):
                                            st.markdown(f"### {line}")
                                        elif line.startswith('Date:') or line.startswith('Time:') or line.startswith('Activity:'):
                                            st.markdown(f"**{line}**")
                                        elif line.startswith('ACTIVE SENSORS:') or line.startswith('LLM ANALYSIS:') or line.startswith('CONTEXT:'):
                                            st.markdown(f"#### {line}")
                                        elif line.startswith('  •') or line.startswith('  -'):
                                            st.markdown(f"{line}")
                                        else:
                                            st.markdown(line)
                            
                            # Show anomaly details in columns
                            if 'anomaly' in result:
                                st.subheader("🔍 Anomaly Details")
                                anomaly_data = result['anomaly']
                                
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Timestamp", pd.to_datetime(anomaly_data.get('Timestamp', 'N/A')).strftime('%Y-%m-%d %H:%M:%S') if pd.notna(anomaly_data.get('Timestamp')) else 'N/A')
                                with col2:
                                    st.metric("Activity", anomaly_data.get('Activity', 'Unknown'))
                                with col3:
                                    st.metric("Confidence", f"{anomaly_data.get('Prediction_Confidence', 0):.2%}")
                        else:
                            st.error("❌ Failed to generate explanation. Please try again.")
                            
                    except Exception as e:
                        st.error(f"❌ Error generating explanation: {str(e)}")
                        with st.expander("Error Details"):
                            st.exception(e)
    
    except Exception as e:
        st.error(f"❌ Error loading anomalies: {str(e)}")
        with st.expander("Error Details"):
            st.exception(e)

else:
    st.info("👈 Please select a date and click 'Find Anomalies' to begin analysis")
    
    # Show some instructions
    st.markdown("""
    ### How to use:
    1. **Select a date** in the sidebar (default: 2011-06-01)
    2. Click **"Find Anomalies"** to see detected anomalies for that date
    3. **Choose an anomaly** from the dropdown list
    4. Click **"Generate LLM Explanation"** to get detailed AI-powered analysis
    
    ### Features:
    - 🔍 View all anomalies detected on a specific date
    - 🤖 Get AI-powered explanations using TinyLlama LLM
    - 📊 See SHAP feature contributions in the analysis
    - 🎯 Understand why each anomaly was flagged as suspicious
    - 📈 View confidence scores and sensor patterns
    """)

# Footer
st.markdown("---")
st.markdown("**IoT Security Anomaly Detection System** | Powered by TinyLlama LLM")
