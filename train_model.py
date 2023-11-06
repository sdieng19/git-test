import os
from flask import Flask, render_template, request
import joblib
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

def read_sensor_data(dir, files):
    dataframes = {}  # Create an empty dictionary to store the DataFrames
    for file in files:
        dataframes[file] = pd.read_csv(os.path.join(dir, file + ".txt"), sep='\t', header=None)
    sensor_data = pd.concat(dataframes.values(), axis=1)
    return sensor_data

def read_valvecondition_data(dir, file):
    data = pd.read_csv(os.path.join(dir, file + ".txt"), sep='\t', header=None)
    # read only valve condition data
    labels = data[[1]]
    labels.loc[labels[1] < 100, "valveCondition"] = 0
    labels.loc[labels[1] == 100, "valveCondition"] = 1
    labels["valveCondition"] = labels["valveCondition"].astype("int64")
    labels = labels[["valveCondition"]]
    print(labels.shape)
    return labels

def predict():
    # Load the pre-trained model or train it and save it to a file
    model_file_path = "model_file.pkl"

    if os.path.exists(model_file_path):
        clf = joblib.load(model_file_path)
    else:
        # Train the model
        features_dir = "./features"
        target_dir = "./target"
        files = ["CE", "CP", "EPS1", "FS1", "FS2", "PS1", "PS2", "PS3", "PS4", "PS5", "PS6", "SE", "TS1", "TS2", "TS3", "TS4", "VS1"]
    
        # Load the data
        data = read_sensor_data(dir=features_dir, files=files)

        # Load the labels/target (assuming it's in a file 'labels.csv')
        labels = read_valvecondition_data(dir=target_dir, file="profile")

        # Standardize the data
        scaler = StandardScaler()
        standardized_data = scaler.fit_transform(data)

        # Apply PCA to reduce dimensionality
        pca = PCA(n_components=200)  # Adjust the number of components
        reduced_data = pca.fit_transform(standardized_data)
        
        # Split the data into training and testing sets
        X_train = reduced_data[:2000]
        y_train = labels.iloc[:2000].values.ravel()
        X_test = reduced_data[2000:]
        y_test = labels.iloc[2000:].values.ravel()

        # Initialize and train the Random Forest classifier
        clf = RandomForestClassifier(verbose=1)
        clf.fit(X_train, y_train)

        # Save the trained model to a file
        joblib.dump(clf, model_file_path)

        data_to_save = {
            "model": clf,
            "reduced_data": reduced_data,
            "labels": labels
        }

        joblib.dump(data_to_save, model_file_path)


if __name__ == '__main__':
    predict()
