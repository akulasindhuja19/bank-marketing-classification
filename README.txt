Bank Marketing Classification Using Machine Learning
Project Overview
This project implements and compares five Machine Learning classification algorithms on the UCI Bank Marketing Dataset to predict whether a customer will subscribe to a bank term deposit.

The objective is to train multiple classifiers, evaluate their performance using standard classification metrics, and identify the most suitable model for this binary classification problem.

The trained models are saved as serialized files and can be deployed through a Streamlit web application for interactive predictions.

Dataset
Dataset Information
This project uses the Bank Marketing "bank-additional-full" dataset from the UCI Machine Learning Repository.

Dataset File:

bank-additional-full.csv
The dataset contains information collected from direct marketing campaigns conducted by a Portuguese banking institution.

The goal is to predict whether a customer will subscribe to a term deposit.

Dataset Summary
Property	Value
Total Records	41,188
Total Columns	21
Input Features	20
Target Variable	y
Positive Class (yes)	4,640
Negative Class (no)	36,548
Missing Values	0
The dataset is imbalanced:

Yes (Subscribed): 11.3%
No (Not Subscribed): 88.7%
Target Variable
The target column is:

y
Possible values:

Value	Meaning
yes	Customer subscribed to a term deposit
no	Customer did not subscribe
Binary encoding used for model training:

y = (df["y"] == "yes").astype(int)
Encoding:

yes → 1
no  → 0
Input Features
The original dataset contains the following features:

Feature
age
job
marital
education
default
housing
loan
contact
month
day_of_week
duration
campaign
pdays
previous
poutcome
emp.var.rate
cons.price.idx
cons.conf.idx
euribor3m
nr.employed
Data Preprocessing
The following preprocessing steps were performed:

1. Load Dataset
df = pd.read_csv("bank-additional-full.csv", sep=";")
2. Remove Data Leakage
The feature:

duration
was removed before training.

Reason:

The duration of the call is only known after the call finishes. Using it during prediction would leak information from the future and artificially improve model performance.

df = df.drop(columns=["duration"])
The final model therefore uses:

19 input features
3. Feature Classification
Numerical and categorical features are automatically separated:

cat_cols = X.select_dtypes(include="object").columns.tolist()
num_cols = X.select_dtypes(exclude="object").columns.tolist()
4. Feature Scaling
Numerical features are standardized using:

StandardScaler()
5. Categorical Encoding
Categorical variables are encoded using:

OneHotEncoder(handle_unknown="ignore")
6. Train-Test Split
The dataset is separated into training and testing data:

train = 80%
test = 20%
using:

train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)
Stratification preserves the original class distribution in both sets.

Machine Learning Models Implemented
The following five classification algorithms were implemented and trained on the same dataset.

1. Logistic Regression
Logistic Regression is a linear model commonly used for binary classification.

LogisticRegression(max_iter=1000)
Advantages
Fast training
Easy interpretation
Strong baseline model
2. Decision Tree Classifier
Decision Trees classify data using hierarchical decision rules.

DecisionTreeClassifier(
    max_depth=8,
    random_state=42
)
Advantages
Highly interpretable
Captures nonlinear relationships
Handles mixed data types
3. K-Nearest Neighbor (KNN)
KNN predicts the class based on neighboring observations.

KNeighborsClassifier(
    n_neighbors=15
)
Advantages
Simple to implement
No explicit training phase
Effective for local decision boundaries
4. Gaussian Naive Bayes
Naive Bayes applies Bayes' theorem while assuming feature independence.

GaussianNB()
Advantages
Very fast
Works well on high-dimensional data
Effective baseline classifier
5. Support Vector Machine (SVM)
SVM finds an optimal decision boundary between classes.

SVC(
    kernel="rbf",
    random_state=42
)
Advantages
Effective in complex feature spaces
Handles nonlinear decision boundaries
Strong classification performance
Evaluation Metrics
Each model is evaluated using six classification metrics.

1. Accuracy
Measures the proportion of correctly classified examples.

Accuracy=
\frac{TP+TN}{TP+TN+FP+FN}
2. AUC Score
Measures the classifier's ability to distinguish between positive and negative classes.

Higher AUC values indicate better class separation.

3. Precision
Measures the correctness of positive predictions.

Precision=
\frac{TP}{TP+FP}
4. Recall
Measures how many actual positive samples are correctly identified.

Recall=
\frac{TP}{TP+FN}
5. F1 Score
Harmonic mean of Precision and Recall.

F1 =
2
\times
\frac{Precision
\times Recall}
{Precision+Recall}
6. Matthews Correlation Coefficient (MCC)
A balanced metric suitable for imbalanced datasets.

MCC =
\frac{TP
\times TN - FP
\times FN}
{
\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}
Project Structure
Bank-Marketing-Classification/
│
├── train_models.py
├── bank-additional-full.csv
├── requirements.txt
├── README.md
├── metrics.csv
├── test_data.csv
│
└── model/
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    └── svm.joblib
Installation
Clone Repository
git clone https://github.com/your-username/bank-marketing-classification.git

cd bank-marketing-classification
Install Dependencies
pip install -r requirements.txt
Running the Project
Run the training script:

python train_models.py bank-additional-full.csv
or

python train_models.py
if the dataset file is already present in the project directory.

Output Files
After successful execution, the following files are generated.

Trained Models
model/logistic_regression.joblib
model/decision_tree.joblib
model/knn.joblib
model/naive_bayes.joblib
model/svm.joblib
Evaluation Metrics
metrics.csv
Contains:

Accuracy
AUC Score
Precision
Recall
F1 Score
MCC
for all five models.

Test Dataset
test_data.csv
Contains the testing split used for evaluating the models and for use in the Streamlit application.

Workflow
Load Bank Marketing Dataset.
Remove the duration feature.
Separate features and target.
Encode categorical variables.
Scale numerical variables.
Split data into training and testing sets.
Train five classification algorithms.
Evaluate model performance.
Save trained models using Joblib.
Export metrics to CSV.
Export testing dataset.
Deploy trained models through Streamlit.
Experimental Results
After execution, the script generates a file named:

metrics.csv
which contains the performance comparison of all implemented classifiers.

ML Model Name	Accuracy	AUC	Precision	Recall	F1 Score	MCC
Logistic Regression	Generated After Execution	Generated After Execution	Generated After Execution	Generated After Execution	Generated After Execution	Generated After Execution
Decision Tree	Generated After Execution	Generated After Execution	Generated After Execution	Generated After Execution	Generated After Execution	Generated After Execution
KNN	Generated After Execution	Generated After Execution	Generated After Execution	Generated After Execution	Generated After Execution	Generated After Execution
Naive Bayes	Generated After Execution	Generated After Execution	Generated After Execution	Generated After Execution	Generated After Execution	Generated After Execution
SVM	Generated After Execution	Generated After Execution	Generated After Execution	Generated After Execution	Generated After Execution	Generated After Execution
Technologies Used
Python 3.x
Pandas
NumPy
Scikit-Learn
Joblib
Streamlit
Conclusion
This project successfully implements and compares five supervised Machine Learning classifiers on the UCI Bank Marketing Dataset. The models are evaluated using Accuracy, AUC Score, Precision, Recall, F1 Score, and Matthews Correlation Coefficient (MCC). The trained models are saved and can be integrated into a Streamlit application for real-time prediction and model comparison.

Author
AKULA SINDHUJA

Machine Learning Lab Assignment

BITS Virtual Lab