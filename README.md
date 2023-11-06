# git-test
# Valve Condition Prediction Flask App

## Overview

This Flask application is designed to predict the condition of hydraulic valves using a pre-trained machine learning model. The model uses sensor data from a hydraulic test rig to make predictions about the condition of the valves. We used a random forest classifier to predict if the condition of the valve is optimal (equal to 100% means 1). If it's not optimal, then the value predicted is 0.  

## Features

- Predict the condition of hydraulic valves based on sensor data.
- Display accuracy and F1 score for individual predictions.
- Access the application via a user-friendly web interface.

## Prerequisites

Before running the application, ensure you have the following dependencies installed:

- Python 3.7+
- Flask (install with `pip install Flask`)
- scikit-learn (install with `pip install scikit-learn`)
- joblib (install with `pip install joblib`)

## Installation

1. Clone the repository to your local machine:

```bash
git clone https://github.com/yourusername/valve-condition-prediction-app.git
```
2. Navigate to the project directory
```bash
cd valve-condition-prediction-app
```

3. Install the required Python packages:
```bash
pip3 install -r requirements.txt
```

## Usage 

1. Train the model by running the train_model.py script. This will generate a pre-trained model file named model_file.pkl.

```bash
python train_model.py
```
2. Start the Flask application:
```bash
python flask_app.py
```

3. Access the application in your web browser at http://localhost:5000.

4. Enter a cycle ID between 2001 and 2204 in the provided form and click "Get Prediction."

5. The application will display the prediction, real label, accuracy, and F1 score for the selected cycle.

6. ps : you can try to select values different from this interval and check the displayed message. 