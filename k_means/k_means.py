import joblib
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans


def create_dataframe(filename):
    """create a dataframe from a dictionary

    Args:
        filename (.pkl): nested dictionary

    Returns:
        pandas dataframe: pandas dataframe
    """
    with open(filename, "rb") as file:
        data_dict = joblib.load(file)
        data_dict.pop('TOTAL', 0)

        df = pd.DataFrame.from_dict(data_dict).T
        df = df.set_index(np.arange(len(data_dict.keys())))
        df = df.replace(['NaN'], np.nan)

    return df


def feature_selector(df, feature_list=None, target=None):
    """create features for ML algorithm from list argument

    Args:
        df (DataFrame): dataframe
        feature_list (list): features columns
        target: target column with binary values.

    Returns:
        tuple: (features Dataframe, target Series)
    """
    features = df[feature_list].fillna(0)
    target = df[target]
    target = pd.get_dummies(target, drop_first=True)

    return (features, target)


def features_plotter(dataframe, nfeatures=2):
    '''plot the features in 2D space to observe clusters'''

    if nfeatures > 2:
        for f1, f2, _ in dataframe.values:
            plt.scatter(f1, f2)
    else:
        for f1, f2 in dataframe.values:
            plt.scatter(f1, f2)

    plt.savefig("datapoints.pdf")


def clustering(features, clusters=3):
    '''perform k-means clustering and return the labels'''

    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(features)

    kmeans = KMeans(n_clusters=clusters)
    kmeans.fit(scaled_features)

    return kmeans.labels_


def draw_clusters(pred, features, poi, mark_poi=False):
    '''draw n clusters on 2D space'''

    colors = ["b", "c", "k", "m", "g"]
    for ii, _ in enumerate(pred):
        plt.scatter(features[ii][0], features[ii]
                    [1], color=colors[pred[ii]])

    if mark_poi:
        for ii, _ in enumerate(pred):
            if poi[ii]:
                plt.scatter(features[ii][0], features[ii]
                            [1], color='r', marker='*')

    plt.xlabel("X-SPACE")
    plt.ylabel("Y-SPACE")
    plt.title(f"Showing {len(np.unique(pred))} Clusters in 2D space")
    plt.savefig(f"show_clusters.pdf")


def main():

    enron_data = create_dataframe("../final_project/final_project_dataset.pkl")

    X, y = feature_selector(df=enron_data, feature_list=[
        "salary", "exercised_stock_options", "total_payments"], target="poi")

    pred = clustering(X, clusters=3)

    features_plotter(dataframe=X, nfeatures=3)
    draw_clusters(pred=pred, features=X.values, poi=y.values, mark_poi=True)


if __name__ == "__main__":
    main()
