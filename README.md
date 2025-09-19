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
│   ├── train.csv           # Training dataset
│   ├── test.csv            # Test dataset (without target)
│   └── sample_submission.csv
├── notebooks/
│   ├── 01_eda.ipynb        # Exploratory Data Analysis
│   ├── 02_preprocessing.ipynb
│   ├── 03_modeling.ipynb
│   └── 04_ensemble.ipynb
├── src/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── models.py
│   └── utils.py
├── submissions/
│   └── final_submission.csv
├── requirements.txt
└── README.md
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

### 🧪 Methodology

#### Data Exploration
- [ ] Analyze distribution of BPM values
- [ ] Explore feature correlations
- [ ] Identify missing values and outliers
- [ ] Visualize audio feature relationships

#### Feature Engineering
- [ ] Create polynomial features
- [ ] Generate interaction terms
- [ ] Apply scaling/normalization
- [ ] Handle categorical variables (if any)

#### Modeling Approach
- [ ] Baseline linear regression
- [ ] Random Forest Regressor
- [ ] Gradient Boosting (XGBoost, LightGBM)
- [ ] Neural Networks
- [ ] Ensemble methods

#### Model Validation
- [ ] K-fold cross-validation
- [ ] Time-based splits (if temporal features exist)
- [ ] Feature importance analysis
- [ ] Hyperparameter tuning

### 📈 Current Results

| Model | CV Score (RMSE) | Public LB Score |
|-------|----------------|-----------------|
| Baseline | - | - |
| Random Forest | - | - |
| XGBoost | - | - |
| Ensemble | - | - |

### 🔧 Key Features and Techniques

- **Feature Engineering**: [Description of key engineered features]
- **Model Selection**: [Rationale for chosen models]
- **Validation Strategy**: [Cross-validation approach]
- **Ensemble Method**: [How models are combined]

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

- pandas
- numpy
- scikit-learn
- xgboost
- lightgbm
- matplotlib
- seaborn
- jupyter

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