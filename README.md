# Explainable Anomaly Detection for Smart Homes (SHAP + TinyLlama)

A Random Forest flags suspicious activity in smart-home sensor data, SHAP shows which sensors drove each flag, and a small local LLM (TinyLlama-1.1B-Chat) turns that into a plain-language explanation. The results are shown in a Streamlit dashboard.

<img width="1916" height="868" alt="Dashboard: anomalies detected on a chosen date" src="https://github.com/user-attachments/assets/740a2326-e486-4b59-9a6c-194e41bab1b6" />

## Results

| Metric (held-out test set, 3,775 one-minute windows) | Value |
| --- | --- |
| F1, anomaly class | 0.80 |
| Precision / recall, anomaly class | 0.81 / 0.79 |
| Accuracy | 98% |

Model: `RandomForestClassifier(n_estimators=200, class_weight="balanced")`, stratified 80/20 split.

## How it works

1. **Data:** the [CASAS Aruba](https://zenodo.org/records/17180309) smart-home dataset. We used the last 100,000 sensor events from 39 motion (M), door (D) and temperature (T) sensors.
2. **Features:** events are aggregated into 18,871 one-minute windows, with one column per sensor plus hour, day of week and a night flag.
3. **Labels:** CASAS has no intrusion labels, so a window is labelled anomalous when there is motion while the annotated activity is *Sleeping* or *Leave_Home*. That gives 1,009 anomalous windows (5.3%).
4. **Detection:** a class-balanced Random Forest predicts anomalies and a confidence score for every window.
5. **Explanation:** for a flagged window, the top 10 SHAP attributions, the active sensors and the time context go into a "security analyst" prompt for TinyLlama-1.1B-Chat, which writes the explanation.
6. **Dashboard:** pick a date, see that day's flagged windows with confidence scores, and generate an explanation for any of them.

<img width="1919" height="875" alt="Dashboard: TinyLlama explanation for a flagged window" src="https://github.com/user-attachments/assets/9daf798d-2776-4ee7-a3e3-60b6e80ea1d5" />

## Dashboard

The Streamlit dashboard is shown in the screenshots above and is not set up as a one-command install. The code is here for reference: `streamlit_app.py` reads the precomputed predictions in `full_pivot_df.pkl` and calls the functions in `anomaly_functions.py`. Rebuilding the predictions means downloading the Aruba data from the link above and running the notebook.

## Repository contents

| File | What it is |
| --- | --- |
| `code_new.ipynb` | Data preparation, labelling, model training, SHAP analysis, prediction export |
| `anomaly_functions.py` | `get_anomalies_by_date` and `explain_single_anomaly` (TinyLlama prompt and generation) |
| `streamlit_app.py` | The dashboard |
| `full_pivot_df.pkl` | Predictions for every window, used by the dashboard |

## Limitations

- The labels come from a rule, not from real intrusions, so the model learns to reproduce that rule from sensor and time features.
- The train/test split is random across consecutive minutes, so neighbouring windows can appear in both sets. A time-based split would give a more conservative estimate.

## Acknowledgements

- [CASAS smart-home datasets](https://zenodo.org/records/17180309), Washington State University
- [TinyLlama](https://github.com/jzhang38/TinyLlama)
- [SHAP](https://github.com/shap/shap)

For educational and research use.
