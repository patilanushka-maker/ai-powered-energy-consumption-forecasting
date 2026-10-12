# Model Artifacts Directory

This directory is designated for serializing and storing trained model artifacts and associated metadata.

---

## Storage Policy

- **Local Generation**: Trained models (`.joblib`, `.pkl`, `.pt`, `.onnx`) are generated locally during the model training phase (`src/train.py`).
- **Version Control Exemption**: Large model binary files are explicitly ignored in `.gitignore` to prevent repository bloat and ensure fast repository cloning.
- **Reproducibility**: Models can be reproduced deterministically by training on the processed data with fixed random seeds defined in configuration files.

---

## Expected Artifacts

When model training is executed in future phases, the following artifacts may reside here:
- Baseline models (e.g., Seasonal Naive, Persistence)
- Machine learning regression models (e.g., Random Forest, Gradient Boosted Trees, Linear Ridge)
- Feature scalers and encoders (`scaler.joblib`)
- Model hyperparameter configurations and training metadata (`model_config.json`)
