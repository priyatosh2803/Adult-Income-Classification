# ML Assignment 2: Classification Models on Adult Income Dataset

## Problem Statement

This project implements multiple machine learning classification models to predict whether an individual's annual income exceeds $50,000 based on demographic and employment-related features. This is a binary classification problem that helps understand the relationship between various socio-economic factors and income levels.

**Objective**: Build and compare 6 different classification models (Logistic Regression, Decision Tree, K-Nearest Neighbors, Naive Bayes, Random Forest, and XGBoost) to predict income class and evaluate their performance using multiple metrics.

---

## Dataset Description

**Dataset**: Adult Income Dataset (also known as "Census Income" dataset)

**Source**: UCI Machine Learning Repository
- **URL**: https://archive.ics.uci.edu/ml/datasets/adult

### Dataset Characteristics:
- **Total Instances**: 48,842 samples (after removing missing values)
- **Number of Features**: 14 features (excluding target)
- **Target Variable**: Binary classification
  - Class 0: Income ≤ $50K
  - Class 1: Income > $50K
- **Feature Types**: Mixed (8 categorical, 6 numerical)

### Feature Description:

| Feature | Type | Description |
|---------|------|-------------|
| age | Continuous | Age of the individual |
| workclass | Categorical | Employment type (Private, Self-emp, Federal-gov, etc.) |
| fnlwgt | Continuous | Final weight (number of people census believes entry represents) |
| education | Categorical | Highest level of education achieved |
| education-num | Continuous | Numerical representation of education level |
| marital-status | Categorical | Marital status (Married-civ-spouse, Never-married, etc.) |
| occupation | Categorical | Type of occupation |
| relationship | Categorical | Relationship status (Husband, Wife, Own-child, etc.) |
| race | Categorical | Race of the individual |
| sex | Categorical | Gender (Male, Female) |
| capital-gain | Continuous | Capital gains |
| capital-loss | Continuous | Capital losses |
| hours-per-week | Continuous | Average hours worked per week |
| native-country | Categorical | Country of origin |

### Data Preprocessing:
- Removed rows with missing values (indicated by '?')
- Encoded categorical variables using Label Encoding
- Applied Standard Scaling to numerical features
- Split data into 80% training and 20% testing sets with stratification

---

## Models Used

### Model Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---------------|----------|-----|-----------|--------|----|----|
| Logistic Regression | 0.8497 | 0.9018 | 0.7584 | 0.6351 | 0.6914 | 0.6152 |
| Decision Tree | 0.8167 | 0.7421 | 0.6535 | 0.5838 | 0.6167 | 0.5082 |
| K-Nearest Neighbors | 0.8384 | 0.8675 | 0.7258 | 0.5925 | 0.6525 | 0.5729 |
| Naive Bayes | 0.8263 | 0.8845 | 0.6951 | 0.6193 | 0.6550 | 0.5464 |
| Random Forest (Ensemble) | 0.8642 | 0.9138 | 0.7889 | 0.6742 | 0.7268 | 0.6542 |
| XGBoost (Ensemble) | 0.8721 | 0.9251 | 0.8045 | 0.6936 | 0.7448 | 0.6798 |

---

## Model Performance Observations

| ML Model Name | Observation about model performance |
|---------------|-------------------------------------|
| **Logistic Regression** | Demonstrates good baseline performance with 84.97% accuracy. The model shows strong AUC (0.9018) indicating excellent discrimination ability between classes. However, precision is higher than recall, suggesting the model is conservative in predicting high-income class. Good balance between interpretability and performance. |
| **Decision Tree** | Shows the lowest performance among all models with 81.67% accuracy and lowest AUC (0.7421). The model tends to overfit training data and generalizes poorly. The relatively balanced precision (0.6535) and recall (0.5838) indicate moderate performance on both classes. Decision boundaries created by the tree are too rigid for this dataset. |
| **K-Nearest Neighbors** | Achieves moderate performance with 83.84% accuracy and good AUC (0.8675). The model performs better than Decision Tree but lacks the robustness of ensemble methods. High precision (0.7258) but lower recall (0.5925) suggests the model is selective in predicting positive class. Performance is sensitive to the choice of k-value and distance metric. |
| **Naive Bayes** | Delivers reasonable performance (82.63% accuracy) considering its simplicity and strong independence assumption. Good AUC score (0.8845) shows effective probability calibration. The balanced precision (0.6951) and recall (0.6193) make it suitable for scenarios where false positives and negatives are equally important. Fast training and prediction times make it practical for large datasets. |
| **Random Forest (Ensemble)** | Shows strong performance with 86.42% accuracy, outperforming all individual classifiers. Excellent AUC (0.9138) and good balance between precision (0.7889) and recall (0.6742). The ensemble approach reduces overfitting seen in individual decision trees. Feature importance from Random Forest provides valuable insights into key income predictors. Robust to outliers and handles non-linear relationships well. |
| **XGBoost (Ensemble)** | Achieves the best overall performance with 87.21% accuracy and highest AUC (0.9251). Superior precision (0.8045) and recall (0.6936) balance demonstrates excellent discrimination ability. The highest MCC score (0.6798) confirms robust performance across both classes. Gradient boosting with regularization prevents overfitting effectively. Handles feature interactions and non-linearities exceptionally well. **Recommended model for deployment** due to superior performance across all metrics. |

### Key Insights:
1. **Ensemble Methods Dominate**: Both Random Forest and XGBoost significantly outperform individual classifiers, with XGBoost achieving the best results.
2. **Trade-off Observation**: Most models show higher precision than recall, indicating they are conservative in predicting the high-income class.
3. **AUC as Key Metric**: All models except Decision Tree achieve AUC > 0.85, showing good class separation capability.
4. **MCC Correlation**: Matthews Correlation Coefficient rankings align well with accuracy rankings, validating consistent performance.
5. **Decision Tree Limitation**: Despite its interpretability, the basic Decision Tree classifier shows clear signs of overfitting and poor generalization.

---

## Project Structure

```
ML_Assignment_2/
│
├── app.py                          # Streamlit web application
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
├── test_data.csv                   # Test dataset for Streamlit app
│
└── model/                          # Model directory
    ├── train_models.py             # Training script for all models
    ├── results.txt                 # Detailed results and metrics
    ├── scaler.pkl                  # Saved StandardScaler
    ├── label_encoders.pkl          # Saved Label Encoders
    ├── logistic_regression_model.pkl
    ├── decision_tree_model.pkl
    ├── k-nearest_neighbors_model.pkl
    ├── naive_bayes_model.pkl
    ├── random_forest_model.pkl
    └── xgboost_model.pkl
```

---

## Installation and Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Step 1: Clone the Repository
```bash
git clone <your-repository-url>
cd ML_Assignment_2
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Train Models
```bash
python model/train_models.py
```

This will:
- Download the Adult Income dataset from UCI repository
- Preprocess the data
- Train all 6 classification models
- Calculate evaluation metrics
- Save trained models in the `model/` directory
- Generate `test_data.csv` for the Streamlit app
- Create `results.txt` with detailed results

### Step 4: Run Streamlit App Locally
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

---

## Streamlit App Features

The interactive web application includes:

1. **📌 Model Selection Dropdown**: Choose from 6 trained classification models
2. **📁 Dataset Upload**: Upload test data in CSV format (with 'income' target column)
3. **📊 Dataset Overview**: Display dataset statistics and sample data
4. **📈 Evaluation Metrics Display**: 
   - Accuracy
   - AUC Score
   - Precision
   - Recall
   - F1 Score
   - Matthews Correlation Coefficient (MCC)
5. **🎯 Confusion Matrix**: Visual representation with heatmap
6. **📋 Classification Report**: Detailed per-class metrics
7. **📊 Prediction Distribution**: Compare actual vs predicted class distributions
8. **💾 Download Results**: Export predictions as CSV file

---

## Deployment on Streamlit Community Cloud

### Prerequisites:
- GitHub account
- Push this repository to GitHub
- Streamlit Community Cloud account (free)

### Deployment Steps:

1. **Push to GitHub**:
   ```bash
   git init
   git add .
   git commit -m "Initial commit: ML Assignment 2"
   git branch -M main
   git remote add origin <your-repo-url>
   git push -u origin main
   ```

2. **Deploy on Streamlit Cloud**:
   - Go to https://streamlit.io/cloud
   - Sign in with your GitHub account
   - Click "New App"
   - Select your repository
   - Choose branch: `main`
   - Select main file path: `app.py`
   - Click "Deploy"

3. **Wait for Deployment**:
   - The app will be built and deployed automatically
   - Usually takes 2-3 minutes
   - You'll receive a public URL: `https://<your-app>.streamlit.app`

---

## Usage Instructions

### For Local Use:
1. Train models using `python model/train_models.py`
2. Run app using `streamlit run app.py`
3. Upload `test_data.csv` or your own test dataset
4. Select a model from the dropdown
5. View metrics, confusion matrix, and classification report
6. Download prediction results

### For Deployed App:
1. Open the Streamlit Cloud URL
2. Upload your test dataset (CSV format)
3. Select desired classification model
4. Analyze results and download predictions

**Note**: Due to Streamlit Cloud's free tier limitations, upload only test data (not the entire dataset).

---

## Requirements

```
streamlit==1.31.0
scikit-learn==1.4.0
numpy==1.26.3
pandas==2.1.4
matplotlib==3.8.2
seaborn==0.13.1
xgboost==2.0.3
```

---

## Model Training Configuration

- **Train-Test Split**: 80-20 ratio with stratification
- **Random State**: 42 (for reproducibility)
- **Cross-validation**: Not applied (can be added for hyperparameter tuning)
- **Hyperparameters**: Default scikit-learn and XGBoost settings

### Model-Specific Configurations:
- **Logistic Regression**: max_iter=1000
- **Decision Tree**: max_depth=10
- **KNN**: n_neighbors=5
- **Naive Bayes**: Gaussian NB (default)
- **Random Forest**: n_estimators=100, max_depth=10
- **XGBoost**: n_estimators=100, max_depth=6

---

## Evaluation Metrics Explained

1. **Accuracy**: Ratio of correct predictions to total predictions
2. **AUC (Area Under ROC Curve)**: Measures model's ability to distinguish between classes
3. **Precision**: Ratio of true positives to total predicted positives
4. **Recall**: Ratio of true positives to total actual positives
5. **F1 Score**: Harmonic mean of precision and recall
6. **MCC (Matthews Correlation Coefficient)**: Correlation between predicted and actual labels (-1 to +1)


---

## Academic Integrity

This project was completed as part of ML Assignment 2 for M.Tech (AIML/DSE) program. All code is original and developed specifically for this assignment. The Adult Income dataset is publicly available from UCI Machine Learning Repository.


---

## Author

**Course**: Machine Learning  
**Program**: M.Tech (AIML)  
**Assignment**: ML Assignment 2  


---

## References

1. UCI Machine Learning Repository: Adult Income Dataset
   - Dua, D. and Graff, C. (2019). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, School of Information and Computer Science.

2. Scikit-learn Documentation: https://scikit-learn.org/
3. XGBoost Documentation: https://xgboost.readthedocs.io/
4. Streamlit Documentation: https://docs.streamlit.io/

---

## Contact

For any queries regarding this project, please contact Priyatosh Sahoo

---
