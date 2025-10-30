# titanic-disaster-prediction-tde
Titanic Disaster Prediction (Python + R Docker Containers)

## Project Overview

This project builds two containerized machine learning pipelines — one in Python and one in R — that predict passenger survival on the Titanic dataset using logistic regression models.
Both implementations follow the same workflow:

Load the Titanic dataset (train.csv, test.csv)

Engineer simple features (FamilySize, IsAlone, Title)

Train a logistic regression model on the training data

Save predictions for the test data to outputs/

## Repository Structure
titanic-disaster-prediction-tde/
├── src/
│   ├── data/                
│   │   ├── train.csv
│   │   ├── test.csv
│   │   └── gender_submission.csv
│   ├── run/                   
│   │   └── model.py
│   └── R_run/                  
│       ├── model.R
│       ├── install_packages.R
│       └── Dockerfile
├── outputs/                   
│   ├── submission_python.csv
│   └── submission_r.csv
├── Dockerfile                
├── requirements.txt          
├── .gitignore
└── README.md

## Data Source

This project uses the Kaggle Titanic dataset

## Download Steps

Go to https://www.kaggle.com/c/titanic/data

Download and unzip the data files.

Move the following CSVs into your local folder:

titanic-disaster-prediction-tde/src/data/


Files required:

train.csv

test.csv


## Requirements

Docker (Desktop or CLI)

No local Python or R setup required

All dependencies install automatically inside each container.

## Running the Python Model
1) Build the Python Docker image

From the project root:

docker build -t titanic-logreg .


This uses the root Dockerfile, installs dependencies from requirements.txt,
and copies your source code into the container.

2) Run the container

If your local data is accessible:

docker run --rm -v "$(pwd)/src/data:/app/src/data" -v "$(pwd)/outputs:/app/outputs" titanic-logreg


If your data was already copied into the image during build:

docker run --rm titanic-logreg


## What happens:

The script trains a logistic regression model.

Prints training accuracy and confusion matrix.

Generates predictions on test.csv.

Saves them to:

outputs/submission_python.csv

## Running the R Model
1) Build the R Docker image
docker build -f src/R_run/Dockerfile -t titanic-r .

2) Run the R container

If your local data is accessible:

docker run --rm -v "$(pwd)/src/data:/app/src/data" -v "$(pwd)/outputs:/app/outputs" titanic-r


If the data was already included in the image:

docker run --rm titanic-r


What happens:

The R script loads the Titanic data.

Engineers features in the same way as the Python version.

Fits a GLM (logistic regression) model.

Saves test predictions to:

outputs/submission_r.csv

## Expected Console Output

Both containers print logs like:

[2025-10-30 14:50:10] Loading file: src/data/train.csv
[2025-10-30 14:50:10] 15) ADD/ADJUST (TRAIN): Creating features (FamilySize, IsAlone, Title).
[2025-10-30 14:50:10] 16) ACCURACY (TRAIN): 0.83
[2025-10-30 14:50:10] 17) PREDICT: Wrote predictions to outputs/submission_python.csv
[2025-10-30 14:50:10] 18) ACCURACY (TEST): Skipped per instructions (save predictions only).
[2025-10-30 14:50:10] DONE.

## Outputs

After running both containers, your folder outputs/ will contain:

submission_python.csv
submission_r.csv


Each file has:

PassengerId	Survived
892	0
893	1
…	…
