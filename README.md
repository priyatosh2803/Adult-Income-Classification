# ML Assignment 2: Classification Models on Adult Income Dataset

## Problem Statement

This project implements multiple machine learning classification models to predict whether an individual's annual income exceeds $50,000 based on demographic and employment-related features. This is a binary classification problem that helps understand the relationship between various socio-economic factors and income levels.

**Objective**: Build and compare 5 different classification models (Logistic Regression, Decision Tree, K-Nearest Neighbors, Naive Bayes, and Support Vector Machine) on the same dataset to predict income class, and evaluate them using Accuracy, AUC, Precision, Recall, F1 Score, and MCC.

---

## Dataset Description

**Dataset**: Adult Income Dataset (also known as "Census Income" dataset)

**Source**: UCI Machine Learning Repository
- **URL**: https://archive.ics.uci.edu/ml/datasets/adult

### Dataset Characteristics:
- **Total Instances**: 48,842 raw rows; 45,222 remain after removing rows with missing values (meets the minimum of 500)
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
| educational-num | Continuous | Numerical representation of education level |
| marital-status | Categorical | Marital status (Married-civ-spouse, Never-married, etc.) |
| occupation | Categorical | Type of occupation |
| relationship | Categorical | Relationship status (Husband, Wife, Own-child, etc.) |
| race | Categorical | Race of the individual |
| gender | Categorical | Gender (Male, Female) |
| capital-gain | Continuous | Capital gains |
| capital-loss | Continuous | Capital losses |
| hours-per-week | Continuous | Average hours worked per week |
| native-country | Categorical | Country of origin |

### Data Preprocessing:
- Stripped leading/trailing whitespace from text columns
- Removed the 3,620 rows containing missing values (indicated by '?')
- Encoded the 8 categorical variables with `OrdinalEncoder` (unseen categories map to -1)
- Applied `StandardScaler` to all features. Encoding and scaling are fit on the training split only, inside a scikit-learn Pipeline, to avoid data leakage
- Split data into 80% training (36,177 rows) and 20% testing (9,045 rows) sets with stratification and `random_state=42`. The test split is saved as `test_data.csv`

---

## GitHub Repository Link

https://github.com/priyatosh2803/Adult-Income-Classification

---

## Live Streamlit App

https://adult-income-classification-wilp-ml-assignment2-2026ak05029.streamlit.app/

---

## Models Used

### Model Comparison Table

Metrics are computed on the 9,045-row held-out test split (`test_data.csv`).

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---------------|----------|-----|-----------|--------|----|-----|
| Logistic Regression | 0.8172 | 0.8473 | 0.7075 | 0.4478 | 0.5485 | 0.4592 |
| Decision Tree | 0.8480 | 0.8931 | 0.7141 | 0.6450 | 0.6778 | 0.5799 |
| K-Nearest Neighbors | 0.8210 | 0.8455 | 0.6597 | 0.5740 | 0.6139 | 0.5001 |
| Naive Bayes | 0.7947 | 0.8478 | 0.6717 | 0.3359 | 0.4478 | 0.3692 |
| Support Vector Machine | 0.8460 | 0.8912 | 0.7574 | 0.5571 | 0.6420 | 0.5572 |

---

## Model Performance Observations

| ML Model Name | Observation about model performance |
|---------------|-------------------------------------|
| **Logistic Regression** | Accuracy 0.8172 and AUC 0.8473. Precision (0.7075) is reasonable, but recall is only 0.4478, so the model misses more than half of the >50K earners. As a linear model it cannot capture the non-linear feature interactions that the tree and SVM exploit. |
| **Decision Tree** | Best on accuracy (0.8480), AUC (0.8931), recall (0.6450), F1 (0.6778) and MCC (0.5799). Its precision (0.7141) and recall are the most balanced of the five. Limiting `max_depth` to 10 keeps the tree from growing too deep. |
| **K-Nearest Neighbors** | Accuracy 0.8210 and AUC 0.8455. Recall (0.5740) trails precision (0.6597). Performance depends on feature scaling and on k=5, which was not tuned. |
| **Naive Bayes** | Lowest accuracy (0.7947) and lowest recall (0.3359), so it misses about two-thirds of the high earners. Gaussian NB assumes independent, normally distributed features, which the skewed capital-gain and capital-loss columns and correlated education features violate. Its AUC (0.8478) is much higher than its recall suggests, so the main weakness is the default decision threshold. |
| **Support Vector Machine** | Accuracy 0.8460 and AUC 0.8912, second only to Decision Tree. Highest precision of all five models (0.7574), so its positive predictions are the most reliable, but recall (0.5571) is lower than the tree's. Uses the default RBF kernel without tuning. |
| **Overall winner for this dataset?** | **Decision Tree** (`max_depth=10`), which ranks first on Accuracy, AUC, Recall, F1 and MCC. SVM is a close second. The gap is small and comes from a single stratified train/test split without cross-validation, so it should not be treated as a definitive ranking. |

### Key Insights:
1. **Decision Tree leads overall**: it ranks first on every metric except precision, where SVM is highest (0.7574).
2. **Precision exceeds recall for every model**: all five are conservative in predicting the high-income class.
3. **AUC is above 0.84 for all models**: each model ranks the two classes reasonably well. The bigger differences lie in recall, which depends on the decision threshold.
4. **MCC agrees with accuracy**: both rank the models in the same order (Decision Tree, SVM, KNN, Logistic Regression, Naive Bayes).
5. **Hyperparameters were not tuned**: the Decision Tree depth limit and the other settings were chosen once, so a tuned model may change the ranking.

---

## Project Structure

```
Adult-Income-Classification/
│
├── app.py                          # Streamlit web application
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
├── adult.csv                       # Full source dataset
├── test_data.csv                   # 20% held-out test split for Streamlit app
│
└── model/                          # Model directory
    ├── train_models.py             # Training and evaluation script
    ├── metrics.csv                 # Metrics for all 5 models
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    └── svm.joblib
```

---

## Installation and Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Step 1: Clone the Repository
```bash
git clone https://github.com/priyatosh2803/Adult-Income-Classification.git
cd Adult-Income-Classification
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
- Load `adult.csv` from the repository root and remove rows with missing values
- Split into 80% train and 20% test (stratified) and save the test split as `test_data.csv`
- Train the 5 classification models inside a preprocessing pipeline
- Save the trained models to `model/*.joblib`
- Calculate evaluation metrics and save them to `model/metrics.csv`

### Step 4: Run Streamlit App Locally
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

---

## Streamlit App Features

The interactive web application includes:

1. **📌 Model Selection Dropdown**: Choose one of the 5 trained classification models
2. **📁 Dataset Upload**: Upload test data in CSV format with the 14 feature columns and the `income` target column. If nothing is uploaded, the bundled `test_data.csv` is used
3. **📊 Model Comparison Table**: Metrics for all 5 models on the test data
4. **📈 Evaluation Metrics Display**: 
   - Accuracy
   - AUC Score
   - Precision
   - Recall
   - F1 Score
   - Matthews Correlation Coefficient (MCC)
5. **🎯 Confusion Matrix**: Actual vs predicted counts for the selected model
6. **📋 Classification Report**: Detailed per-class metrics

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
   git remote add origin https://github.com/priyatosh2803/Adult-Income-Classification.git
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

### For Deployed App:
1. Open the Streamlit Cloud URL
2. Upload your test dataset (CSV format)
3. Select desired classification model
4. Analyze results for each model

**Note**: Due to Streamlit Cloud's free tier limitations, upload only test data (not the entire dataset).

---

## Requirements

```
streamlit==1.66.1
scikit-learn==1.9.1
pandas==3.1.0
numpy==2.4.6
joblib==1.6.0
```

---

## Model Training Configuration

- **Train-Test Split**: 80-20 ratio with stratification
- **Random State**: 42 (for reproducibility)
- **Cross-validation**: Not applied (can be added for hyperparameter tuning)
- **Hyperparameters**: Default scikit-learn settings unless listed below

### Model-Specific Configurations:
- **Logistic Regression**: max_iter=1000
- **Decision Tree**: max_depth=10
- **KNN**: n_neighbors=5
- **Naive Bayes**: Gaussian NB (default)
- **Support Vector Machine**: kernel='rbf' (default C=1.0, gamma='scale')

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

This project was completed as part of ML Assignment 2 for M.Tech AIML program. All code is original and developed specifically for this assignment. The Adult Income dataset is publicly available from UCI Machine Learning Repository.


---

## Author

**Name**: Priyatosh Sahoo
**Course**: Machine Learning  
**Program**: M.Tech (AIML)  
**Assignment**: ML Assignment 2  


---

## References

1. UCI Machine Learning Repository: Adult Income Dataset
   - Dua, D. and Graff, C. (2019). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, School of Information and Computer Science.

2. Scikit-learn Documentation: https://scikit-learn.org/
3. Streamlit Documentation: https://docs.streamlit.io/

---

## Contact

For any queries regarding this project, please contact Priyatosh Sahoo

---
