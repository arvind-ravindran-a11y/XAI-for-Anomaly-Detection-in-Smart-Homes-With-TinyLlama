"""
Anomaly Analysis Functions for Streamlit App
Extracted from the notebook for use in Streamlit
"""

import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


def get_anomalies_by_date(date_input, prediction_df=None, feature_cols=None):
    """
    Get all anomalies detected on a specific date with their features.
    
    Parameters:
    -----------
    date_input : str or datetime
        Date to query (format: 'YYYY-MM-DD' or datetime object)
    prediction_df : DataFrame
        DataFrame with predictions and timestamps
    feature_cols : list
        List of feature column names to include (default: None, uses all X columns)
    
    Returns:
    --------
    DataFrame with anomalies for the specified date
    """
    if prediction_df is None:
        raise ValueError("prediction_df is required. Please provide full_pivot_df or test_pivot_df.")
    
    # Convert date input to datetime if string
    if isinstance(date_input, str):
        query_date = pd.to_datetime(date_input).date()
    else:
        query_date = date_input.date() if hasattr(date_input, 'date') else date_input
    
    # Filter for the specified date and anomalies only
    prediction_df = prediction_df.copy()
    prediction_df['Date'] = prediction_df['Timestamp'].dt.date
    anomalies = prediction_df[
        (prediction_df['Date'] == query_date) & 
        (prediction_df['Predicted_Intrusion'] == 1)
    ].copy()
    
    if len(anomalies) == 0:
        return None
    
    # Select relevant columns
    if feature_cols is None:
        # Get feature columns (exclude metadata columns)
        metadata_cols = ['Timestamp', 'Date', 'Activity', 'Intrusion', 'Predicted_Intrusion', 
                        'Prediction_Confidence', 'Hour', 'DayOfWeek', 'IsNight']
        feature_cols = [col for col in prediction_df.columns if col not in metadata_cols]
    
    # Select columns to return
    result_cols = ['Timestamp', 'Activity', 'Predicted_Intrusion', 'Prediction_Confidence'] + feature_cols
    result_cols = [col for col in result_cols if col in anomalies.columns]
    
    return anomalies[result_cols].sort_values('Timestamp')


def explain_single_anomaly(date_input, anomaly_index, prediction_df=None,
                           model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                           shap_values=None, X=None, explainer=None, scaler=None):
    """
    Explain a single specific anomaly using TinyLlama model.
    
    Parameters:
    -----------
    date_input : str or datetime
        Date to query (format: 'YYYY-MM-DD' or datetime object)
    anomaly_index : int
        Index of the anomaly to explain (0-based, from the list of anomalies)
    prediction_df : DataFrame
        DataFrame with predictions (default: full_pivot_df)
    model_name : str
        HuggingFace model name for local LLM (default: "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    shap_values : array, optional
        Pre-computed SHAP values
    X : DataFrame, optional
        Feature DataFrame for SHAP mapping
    explainer : object, optional
        SHAP explainer for on-the-fly calculation
    scaler : object, optional
        Scaler used for feature scaling
    
    Returns:
    --------
    dict: Dictionary containing the anomaly details and LLM explanation
    """
    if prediction_df is None:
        raise ValueError("prediction_df is required. Please provide full_pivot_df or test_pivot_df.")
    
    anomalies = get_anomalies_by_date(date_input, prediction_df)
    
    if anomalies is None or len(anomalies) == 0:
        return {
            'error': f"No anomalies found on {date_input}",
            'anomaly': None,
            'explanation': None
        }
    
    if anomaly_index < 0 or anomaly_index >= len(anomalies):
        return {
            'error': f"Invalid anomaly index. Must be between 0 and {len(anomalies)-1}",
            'anomaly': None,
            'explanation': None
        }
    
    anomaly = anomalies.iloc[anomaly_index]
    
    date_str = str(date_input) if not isinstance(date_input, str) else date_input
    timestamp = anomaly['Timestamp']
    activity = anomaly.get('Activity', 'Unknown')
    confidence = anomaly.get('Prediction_Confidence', 0)
    hour = pd.to_datetime(timestamp).hour if pd.notna(timestamp) else 'Unknown'
    
    metadata_cols = ['Timestamp', 'Date', 'Activity', 'Intrusion', 'Predicted_Intrusion', 
                    'Prediction_Confidence', 'Hour', 'DayOfWeek', 'IsNight']
    feature_cols = [col for col in anomaly.index if col not in metadata_cols]
    
    active_sensors = []
    for col in feature_cols:
        try:
            if col in anomaly.index:
                value = anomaly[col]
                if isinstance(value, pd.Series):
                    value = value.iloc[0] if len(value) > 0 else 0
                if hasattr(value, 'item'):
                    try:
                        value = value.item()
                    except (ValueError, AttributeError):
                        pass
            else:
                value = 0
        except (KeyError, IndexError, AttributeError):
            value = 0
        
        try:
            if isinstance(value, (pd.Series, list, np.ndarray)):
                value = value[0] if len(value) > 0 else 0
            
            if pd.notna(value) and value != 0:
                if col.startswith('M'):
                    active_sensors.append(f"{col}: Motion detected")
                elif col.startswith('D'):
                    active_sensors.append(f"{col}: {'Open' if value > 0 else 'Closed'}")
                elif col.startswith('T'):
                    active_sensors.append(f"{col}: {value}°C")
                else:
                    active_sensors.append(f"{col}: {value}")
        except (ValueError, TypeError, AttributeError):
            pass
    
    motion_detected = any('M' in s for s in active_sensors)
    door_activity = any('D' in s for s in active_sensors)
    is_night = hour >= 22 or hour < 6
    
    # Get SHAP values
    shap_info = ""
    try:
        if shap_values is not None and X is not None:
            anomaly_idx_in_full = anomaly.name if hasattr(anomaly, 'name') else None
            if anomaly_idx_in_full is not None:
                try:
                    test_idx = list(X.index).index(anomaly_idx_in_full)
                    if len(shap_values.shape) == 3:
                        instance_shap = shap_values[test_idx, :, 1]
                    else:
                        instance_shap = shap_values[test_idx, :]
                    
                    shap_dict = dict(zip(X.columns, instance_shap))
                    top_shap = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
                    
                    shap_lines = []
                    for feat, val in top_shap:
                        direction = "increases" if val > 0 else "decreases"
                        shap_lines.append(f"  - {feat}: {val:.4f} ({direction} anomaly probability)")
                    
                    if shap_lines:
                        shap_info = "\nSHAP Feature Contributions (Top 10):\n" + "\n".join(shap_lines)
                except (ValueError, IndexError):
                    pass
        
        if not shap_info and explainer is not None and X is not None and scaler is not None:
            feature_values = []
            for col in X.columns:
                if col in anomaly.index:
                    val = anomaly[col]
                    if isinstance(val, pd.Series):
                        val = val.iloc[0] if len(val) > 0 else 0
                    if hasattr(val, 'item'):
                        try:
                            val = val.item()
                        except:
                            pass
                    feature_values.append(float(val) if pd.notna(val) else 0.0)
                else:
                    feature_values.append(0.0)
            
            if len(feature_values) > 0:
                feature_array = np.array(feature_values).reshape(1, -1)
                feature_scaled = scaler.transform(feature_array)
                instance_shap = explainer.shap_values(feature_scaled)
                
                if len(instance_shap.shape) == 3:
                    instance_shap = instance_shap[0, :, 1]
                else:
                    instance_shap = instance_shap[0, :]
                
                shap_dict = dict(zip(X.columns, instance_shap))
                top_shap = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
                
                shap_lines = []
                for feat, val in top_shap:
                    direction = "increases" if val > 0 else "decreases"
                    shap_lines.append(f"  - {feat}: {val:.4f} ({direction} anomaly probability)")
                
                if shap_lines:
                    shap_info = "\nSHAP Feature Contributions (Top 10):\n" + "\n".join(shap_lines)
    except Exception:
        shap_info = ""
    
    # Load and use TinyLlama model
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True
    )
    
    model.eval()
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    sensor_info = "\n".join([f"  - {s}" for s in active_sensors]) if active_sensors else "  - No active sensors detected"
    
    prompt = f"""<|system|>
You are an expert security analyst. Analyze IoT sensor anomalies clearly and concisely.
<|user|>
Security Anomaly Analysis:

Date: {date_str}
Time: {timestamp}
Activity: {activity}
Confidence: {confidence:.2%}
Hour: {hour}:00 ({'Night' if is_night else 'Day'} hours)

Active Sensors:
{sensor_info}
{shap_info}

Context:
- Expected activity: {activity}
- Time of day: {'Night (22:00-06:00)' if is_night else 'Day (06:00-22:00)'}
- Motion sensors: {'Active' if motion_detected else 'Inactive'}
- Door sensors: {'Active' if door_activity else 'Inactive'}

Explain:
1. When this anomaly occurred
2. Why it was flagged as suspicious
3. Which sensor patterns triggered detection
4. Security implications
5. Recommendations

Focus on the SHAP values to explain which features contributed most to the anomaly classification.
<|assistant|>
"""
    
    inputs = tokenizer.encode(prompt, return_tensors="pt", max_length=512, truncation=True)
    
    if torch.cuda.is_available():
        inputs = inputs.to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=800,
            temperature=0.8,
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1,
            no_repeat_ngram_size=2
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    input_text = tokenizer.decode(inputs[0], skip_special_tokens=True)
    
    if "<|assistant|>" in generated_text:
        llm_explanation = generated_text.split("<|assistant|>")[-1].strip()
    else:
        llm_explanation = generated_text[len(input_text):].strip()
    
    if not llm_explanation or len(llm_explanation) < 10:
        if "<|user|>" in generated_text:
            parts = generated_text.split("<|user|>")
            if len(parts) > 1:
                llm_explanation = parts[-1].split("<|assistant|>")[-1].strip() if "<|assistant|>" in parts[-1] else ""
    
    llm_explanation = llm_explanation.lstrip('?').strip()
    
    lines = llm_explanation.split('\n')
    cleaned_lines = []
    for i, line in enumerate(lines):
        line_stripped = line.strip()
        if any(line_stripped.startswith(prefix) for prefix in ['4. How does', '5. Are there', 'Please provide', 'Provide a professional']):
            continue
        if not cleaned_lines and not line_stripped:
            continue
        cleaned_lines.append(line)
    
    llm_explanation = '\n'.join(cleaned_lines).strip()
    
    if not llm_explanation or len(llm_explanation) < 20:
        llm_explanation = "Analysis could not be generated. Please check model configuration."
    
    explanation = f"""ANOMALY ANALYSIS REPORT
Date: {date_str}
Time: {timestamp}
Activity: {activity}
Confidence: {confidence:.2%}
Hour: {hour}:00 ({'Night' if is_night else 'Day'} hours)

ACTIVE SENSORS:
{chr(10).join(f'  • {s}' for s in active_sensors) if active_sensors else '  • No active sensors detected'}

LLM ANALYSIS:
{llm_explanation}

CONTEXT:
- Expected activity: {activity}
- Time of day: {'Night (22:00-06:00)' if is_night else 'Day (06:00-22:00)'}
- Motion sensors: {'Active' if motion_detected else 'Inactive'}
- Door sensors: {'Active' if door_activity else 'Inactive'}"""
    
    return {
        'date': date_input,
        'anomaly_index': anomaly_index,
        'anomaly': anomaly.to_dict(),
        'explanation': explanation
    }



