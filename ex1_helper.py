import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def create_model(learning_rate, layers):
    model = keras.Sequential([
        keras.layers.Dense(1, input_shape=(layers,))
    ])
    model.compile(
        optimizer=keras.optimizers.SGD(learning_rate=learning_rate),
        loss='mse'
    )
    return model

def show_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    print(hist)

def transform_data(X, y):
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=3103)

    # Scale
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test

def mask_df(df, feature_cols, target_col):
    mask = df['MedHouseVal'] < 5
    X_filtered = df.loc[mask, feature_cols]
    y_filtered = df.loc[mask, target_col]
    return X_filtered, y_filtered

def plot_heatmap(df, title):
    corr_matrix = df.corr()
    fig, ax = plt.subplots(figsize=(10, 10))
    cax = ax.matshow(corr_matrix, cmap='coolwarm')
    fig.colorbar(cax)

    ax.set_xticks(np.arange(len(df.columns)))
    ax.set_yticks(np.arange(len(df.columns)))
    ax.set_xticklabels(df.columns, rotation=90)
    ax.set_yticklabels(df.columns)

    plt.title(title, pad=5)
    plt.show()

def calculate_stats(features):
    feat_mean = round(np.mean(features),2)
    feat_median = round(np.median(features),2)
    feat_std = round(np.std(features),2)
    print(f'MedInc: Mean {feat_mean}. Median {feat_median}. Std {feat_std}')
    return feat_mean, feat_median, feat_std