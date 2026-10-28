1. Code
1_fire_detect_attribute.py: Downloads the preliminarily screened active-fire candidates.
2_extract_neighborhood_bands.py: Extracts/downloads all required spatial-spectral features for the active fires.
3_train_xgboost.py: Trains the XGBoost classification model.
4_bright_roof_timeseries_mask_batch.py: Generates the persistent bright-object mask (PBOM).
5_predict_timeseries_mask_fire_only.py: Outputs the final detected active fires and other identified noise/false alarms.

2. Data
2023...T15QYB...: Sample data used for testing.

3. model
20260316_1.pkl: the final trained model used in the manuscript.
