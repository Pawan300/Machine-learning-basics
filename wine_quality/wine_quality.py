import os
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt


def loading_data(path):
    if os.path.isfile(path):
        data = pd.read_csv(path, sep=';')
        return data, True
    
    return [], False

def plot_corr(data):

    plt.figure(figsize=(12, 12))
    sns.heatmap(data, annot=True)
    plt.savefig("Images/corr.png")

def creating_class(data):

    label = []
    for i in data["quality"]:

        if i<6: label.append(-1)
        elif i>6: label.append(1)
        else : label.append(0)

    data["quality"] = label
    return data

def analysis(data):

    print("\nData Shape : ", data.shape)
    print("\nColumns : ", data.columns)

    print("\nDropping PH columns beacuse of high correlation.")

    data = data.drop("pH", axis=1)

    print("\nClass Distibution : \n")
    print(data["quality"].value_counts())

    data = creating_class(data)
    return data

def train_test(data):

    y = data["quality"]
    data = data.drop("quality", axis=1)
    train, test, y_train, y_test = train_test_split(data, y, test_size=0.25)

    return train, test, y_train, y_test

def main():

    data, status = loading_data("winequality-red.csv")
    if not status : return "Path not found"

    analysis(data)
    train, test, y_train, y_test = train_test(data)

    print("\n\nTrain shape : ", train.shape)
    print("Test shape : ", test.shape, end="\n\n\n")

    print("*"*50,"\nModeling : ")

    rc = RandomForestClassifier(n_jobs=-1)
    rc.fit(train, y_train)
    y_pred = rc.predict(test)
    
    # Evalute model predictions using precision and recall
    precision = precision_score(y_test, y_pred, average="micro")
    recall = recall_score(y_test, y_pred, average="micro")
    f1 = f1_score(y_test, y_pred, average="micro")
    print('F1_score : {} / Precision: {} / Recall: {}\n\n'.format(round(f1, 3), round(precision, 3), round(recall, 3)))
    
    print("*"*50,"Grid_search : ")
    param_grid = {
        "n_estimators" : [100, 150, 200, 300],
        "max_depth" : [None, 2,3,4],
        "criterion" : ["gini", "entropy"]
    }
    f1 = make_scorer(f1_score , average='micro')
    model = GridSearchCV(
        estimator=rc,
        param_grid = param_grid,
        scoring = f1,
        verbose = 10,
        n_jobs = 1,
        cv=5
    )

    model.fit(train, y_train)

    print("Best Score : ", model.best_score_)
    print("Best parameter : ", model.best_estimator_.get_params())

    return "Completed Successfully"


if __name__ == "__main__":
    print(main())