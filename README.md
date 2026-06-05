-Student Performance Predictor

+ A simple Machine Learning project using PyTorch, Pandas, and Scikit-Learn to predict whether a student will pass or fail based on study habits and lifestyle factors.

-FEATURES: 

This project uses the following student attributes:

1.Study Hours
2.Attendance
3.Sleep Hours
4.Phone Usage Time
5.Stress Level

A feed-forward neural network is trained to classify whether a student passes or fails.

-Project Structure:

Student_Performance_Real_Data/
│
├── Student_Performance_Import_Pandas.py
├── student_performance_ml.csv
└── README.md

-REQUIREMENTS: 

+Python Version

This project was developed with: Python 3.1x

You can check your Python version using:
"python --version" in your terminal

INSTALLATION:
1. Clone the repository using bash: 

"git clone <YOUR_REPOSITORY_URL>"
"cd Student_Performance_Real_Data"

2. Create a virtual environment (recommended)

Windows bash:

"python -m venv venv"
"venv\Scripts\activate"

Mac/Linux bash:
"python3 -m venv venv"
"source venv/bin/activate"

3. Install dependencies using bash:
"pip install pandas numpy torch scikit-learn matplotlib"

Or install all packages at once:

bash :
"python -m pip install pandas numpy torch scikit-learn matplotlib"

DATASET:

The dataset file: "student_performance_ml.csv"

must remain in the SAME folder as "Student_Performance_Import_Pandas.py"

NOTE: The script automatically loads the CSV file using a relative path.

RUNNING THE PROJECT: 
Execute:

bash: "python Student_Performance_Import_Pandas.py"

The program will:

1. Load the dataset
2. Split data into train/test sets
3. Normalize features using StandardScaler
4. Train a PyTorch neural network
5. Evaluate model performance
6. Print classification metrics
7. Display training and validation loss curves

MODEL ARCHITECTURE: 

-Input Features: 5

-Neural Network Layers: 5 → 16 → 8 → 4 → 1

-Activation Function: ReLU()

-Loss Function: BCEWithLogitsLoss()

-Optimizer: Adam

-Learning Rate: 0.001

-Epochs: 1000

-Classification Threshold: 0.7

-Evaluation Metrics

The script reports:

* Accuracy
* Confusion Matrix
* Precision
* Recall
* F1 Score
* Classification Report

Example output:

Accuracy: 0.92

Confusion Matrix:
[[88  5]
 [ 7 100]]

Precision: 0.95
Recall: 0.93
F1 Score: 0.94

(Note: Results may vary depending on dataset and random seed.)

VISUALIZATION: 

-After training, the program displays:
1.Training Loss
2.Validation Loss

This helps monitor model convergence and detect overfitting.

TECHNOLOGIES USED:

1.Python
2.Pandas
3.NumPy
4.PyTorch
5.Scikit-Learn
6.Matplotlib

PURPOSES:
-Created as a learning project for practicing:

1. Data preprocessing
2. Binary classification
3. Neural networks with PyTorch
4. Model evaluation
5. Machine Learning workflow

FUTURE DEVELOPMENT FOR THE PROJECT:
1. Hyperparameter tuning
2. K-Fold Cross Validation
3. Early Stopping
4. Model Saving and Loading
5. Deployment with Flask/FastAPI
6. Real-world student dataset collection