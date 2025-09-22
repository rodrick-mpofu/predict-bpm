# Predicting the Beats-per-Minute of Songs

## Kaggle Playground Series - Season 5, Episode 9

### 🎵 Project Overview

This repository contains my solution for the Kaggle Playground Series competition focused on predicting the beats-per-minute (BPM) of songs using machine learning techniques. The goal is to build a regression model that accurately predicts a song's tempo from various audio features.

### 🎯 Competition Details

- **Competition Type**: Regression (Continuous Target Variable)
- **Evaluation Metric**: Root Mean Squared Error (RMSE)
- **Timeline**: September 1-30, 2025
- **Dataset**: Synthetically generated from real-world music data

### 📊 Dataset Information

The dataset contains various audio features that can be used to predict a song's BPM. This is a synthetic dataset created from real-world music data, designed to provide realistic patterns while ensuring test labels remain private.

**Target Variable**: `BeatsPerMinute` - Continuous values representing song tempo

### 🏆 Competition Goals

- Predict continuous BPM values for songs in the test set
- Minimize Root Mean Squared Error between predictions and actual BPM
- Practice regression techniques and feature engineering
- Explore audio feature relationships with song tempo

### 📁 Repository Structure

```
├── data/
│   ├── train.csv           # Training dataset with BPM targets
│   ├── test.csv            # Test dataset (without target)
│   └── sample_submission.csv # Competition submission format
├── notebooks/
│   └── 01_eda.ipynb        # Exploratory Data Analysis ✅
├── src/
│   ├── __init__.py         # Package initialization
│   ├── data_preprocessing.py # Data cleaning and preprocessing ✅
│   └── utils.py            # Utility functions and helpers ✅
├── submissions/
│   └── ensemble_submission.csv # Generated predictions
├── .gitignore              # Git ignore rules for ML projects ✅
├── requirements.txt        # Python dependencies ✅
└── README.md              # Project documentation
```

### 🛠️ Setup and Installation

1. Clone this repository:
```bash
git clone https://github.com/[username]/predict-bpm.git
cd predict-bpm
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Download the competition data from Kaggle and place in the `data/` folder.

### 🧪 Methodology & Progress

#### Data Exploration ✅
- [x] Analyze distribution of BPM values (completed in EDA notebook)
- [x] Explore feature correlations and relationships
- [x] Identify missing values and outliers
- [x] Visualize audio feature relationships with target
- [x] Statistical analysis of feature distributions

#### Feature Engineering 🔄
- [x] Data preprocessing pipeline implemented
- [x] Missing value handling strategies
- [x] Feature scaling and normalization
- [ ] Create polynomial features
- [ ] Generate interaction terms
- [ ] Domain-specific audio feature engineering

#### Modeling Approach 📋
- [ ] Baseline linear regression
- [ ] Random Forest Regressor
- [ ] Gradient Boosting (XGBoost, LightGBM, CatBoost)
- [ ] Neural Networks (if beneficial)
- [ ] Ensemble methods for final predictions

#### Model Validation 📋
- [ ] K-fold cross-validation setup
- [ ] Feature importance analysis
- [ ] Hyperparameter tuning with Optuna
- [ ] Model interpretability with SHAP

### 📈 Current Results

| Model | CV Score (RMSE) | Public LB Score |
|-------|----------------|-----------------|
| Baseline | - | - |
| Random Forest | - | - |
| XGBoost | - | - |
| Ensemble | - | - |

### 🔧 Technical Implementation Details

#### Current Architecture
- **Data Pipeline**: Modular preprocessing with `data_preprocessing.py`
- **Utility Functions**: Reusable components in `utils.py` for model evaluation and visualization
- **Notebook-Based EDA**: Comprehensive exploratory analysis in Jupyter notebooks

#### Feature Engineering Strategy
- **Missing Value Handling**: Statistical imputation based on feature distributions
- **Scaling Strategy**: StandardScaler for continuous features, preserving audio feature relationships
- **Feature Selection**: Correlation analysis and domain knowledge for audio features
- **Future Plans**: Polynomial features, interaction terms, and domain-specific transformations

#### Model Development Approach
- **Baseline Strategy**: Start with linear regression for interpretability
- **Tree-Based Models**: Random Forest and Gradient Boosting (XGBoost, LightGBM, CatBoost)
- **Ensemble Strategy**: Weighted averaging and stacking approaches
- **Validation**: Stratified K-fold cross-validation to ensure robust performance estimates

#### Technical Stack
- **Core ML**: scikit-learn ecosystem with gradient boosting libraries
- **Hyperparameter Optimization**: Optuna for efficient parameter search
- **Model Interpretation**: SHAP values for feature importance and model explainability
- **Visualization**: matplotlib/seaborn for EDA, plotly for interactive plots

### 📋 Submission Format

Predictions should be submitted in CSV format:
```csv
ID,BeatsPerMinute
524164,119.5
524165,127.42
524166,111.11
```

### 🚀 How to Run

1. **Exploratory Data Analysis**:
   ```bash
   jupyter notebook notebooks/01_eda.ipynb
   ```

2. **Training Models**:
   ```bash
   python src/models.py
   ```

3. **Generate Predictions**:
   ```bash
   python src/predict.py
   ```

### 📚 Dependencies

#### Core Data Science Stack
- **pandas** (≥1.5.0) - Data manipulation and analysis
- **numpy** (≥1.21.0) - Numerical computing
- **scikit-learn** (≥1.1.0) - Machine learning algorithms and preprocessing
- **scipy** (≥1.8.0) - Statistical functions
- **statsmodels** (≥0.13.0) - Advanced statistical analysis

#### Machine Learning Libraries
- **xgboost** (≥1.6.0) - Gradient boosting framework
- **lightgbm** (≥3.3.0) - Fast gradient boosting
- **catboost** (≥1.1.0) - Categorical feature handling
- **optuna** (≥3.0.0) - Hyperparameter optimization

#### Visualization & Analysis
- **matplotlib** (≥3.5.0) - Static plotting
- **seaborn** (≥0.11.0) - Statistical visualization
- **plotly** (≥5.10.0) - Interactive visualizations
- **shap** (≥0.41.0) - Model interpretation

#### Development Tools
- **jupyter** (≥1.0.0) - Interactive notebooks
- **tqdm** (≥4.64.0) - Progress bars
- **joblib** (≥1.1.0) - Model persistence
- **feature-engine** (≥1.5.0) - Feature engineering utilities

*See `requirements.txt` for complete dependency list with version specifications.*

### 🤝 Competition Context

This competition is part of the Kaggle Tabular Playground Series, designed to provide:
- Lightweight challenges for skill development
- Synthetic datasets based on real-world data
- Opportunities for rapid experimentation
- Community learning and collaboration

### 📝 Notes

- The dataset is synthetically generated but maintains realistic patterns from actual music data
- Focus on regression techniques and feature engineering
- RMSE optimization is key to achieving good performance
- Cross-validation is crucial for reliable model evaluation

### 🎖️ Competition Prizes

- 1st-3rd Place: Choice of Kaggle merchandise
- Focus on learning and community participation

### 📄 Citation

Walter Reade and Elizabeth Park. Predicting the Beats-per-Minute of Songs. https://kaggle.com/competitions/playground-series-s5e9, 2025. Kaggle.

---

### 📞 Contact

Feel free to reach out for discussions about approaches, feature engineering ideas, or collaboration opportunities!

**Happy Modeling! 🎵📊**