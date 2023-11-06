from flask import Flask, render_template, request
import pandas as pd
import joblib
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

app = Flask(__name__)

# Define a global variable for reduced_data
reduced_data = None

# Define a global variable for the trained model
clf = None


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

def train_model():
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
    pca = PCA(n_components=100)  
    reduced_data = pca.fit_transform(standardized_data)

    # Split the data into training and testing sets
    X_train = reduced_data[:2000]
    y_train = labels.iloc[:2000].values.ravel()

    # Initialize and train the Random Forest classifier
    clf = RandomForestClassifier(verbose=1)
    clf.fit(X_train, y_train)

    return reduced_data, labels


@app.route("/", methods=["GET", "POST"])
def predict():
    
    global clf
    global reduced_data
    global labels

    # Load the model and associated data
    loaded_data = joblib.load("model_file.pkl")

    if loaded_data is None:
        reduced_data, labels = train_model()
    else:
        # Load the pre-trained model
        # Extract the model, reduced_data, and labels
        clf = loaded_data["model"]
        reduced_data = loaded_data["reduced_data"]
        labels = loaded_data["labels"]
        
        accuracy = None
        f1 = None

        if request.method == "POST":
            cycle_id = int(request.form["cycle_id"])
            if 2001 <= cycle_id <= 2204:
                cycle_data = reduced_data[cycle_id]
                prediction = clf.predict([cycle_data])[0]
                real_label = labels.iloc[cycle_id][0]
                
                # Calculate accuracy and F1 score for this specific prediction
                y_test = labels.iloc[cycle_id][0]
                y_pred = prediction
                accuracy = accuracy_score([y_test], [y_pred])
                f1 = f1_score([y_test], [y_pred])

                return render_template("index.html", prediction=prediction, real_label=real_label, accuracy=accuracy, f1=f1)
            
            else:
                return render_template("index.html", error="Invalid cycle ID. Please provide a valid ID between 2001 and 2204.")
    return render_template("index.html")




if __name__ == '__main__':
    app.run(host="0.0.0.0", port=82)

