import pandas as pd

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

def load_data():
    # Set paths
    PATH = "resources/UCI HAR Dataset/"
    features_path = PATH + "features.txt"
    activity_labels_path = PATH + "activity_labels.txt"
    X_train_path = PATH + "train/X_train.txt"
    y_train_path = PATH + "train/y_train.txt"
    X_test_path = PATH + "test/X_test.txt"
    y_test_path = PATH + "test/y_test.txt"

    # Load feature names
    features_df = pd.read_csv(features_path, sep="\\s+", header=None, names=["idx", "feature"])
    feature_names = features_df["feature"].tolist()
    unique_feature_names = make_unique(feature_names)  # Assuming make_unique is defined

    # Load activity labels (mapping IDs 1-6 to string names)
    activity_labels_df = pd.read_csv(activity_labels_path, sep="\\s+", header=None, names=["id", "activity"])
    activity_map = dict(zip(activity_labels_df.id, activity_labels_df.activity))

    # Load feature data and activity labels for train/test sets
    X_train = pd.read_csv(X_train_path, sep="\\s+", header=None, names=unique_feature_names)
    y_train = pd.read_csv(y_train_path, sep="\\s+", header=None, names=["Activity"])
    X_test = pd.read_csv(X_test_path, sep="\\s+", header=None, names=unique_feature_names)
    y_test = pd.read_csv(y_test_path, sep="\\s+", header=None, names=["Activity"])

    # Map the activity IDs to their string names
    y_train = y_train["Activity"].map(activity_map)
    y_test = y_test["Activity"].map(activity_map)
    
    return X_train, X_test, y_train, y_test

def separate_subjects(X_train, X_test, y_train, y_test):
    # Load subject data
    PATH = "resources/UCI HAR Dataset/"
    subject_train_path = PATH + "train/subject_train.txt"
    subject_test_path  = PATH + "test/subject_test.txt"
    subject_train = pd.read_csv(subject_train_path, header=None, names=["Subject"])
    subject_test  = pd.read_csv(subject_test_path, header=None, names=["Subject"])

    # Combine the training and test sets
    X = pd.concat([X_train, X_test], axis=0).reset_index(drop=True)
    y = pd.concat([y_train, y_test], axis=0).reset_index(drop=True)
    subjects = pd.concat([subject_train, subject_test], axis=0).reset_index(drop=True)

    # Perform a subject-wise split using GroupShuffleSplit
    gss = GroupShuffleSplit(n_splits=1, random_state=3103)
    train_idx, test_idx = next(gss.split(X, y, groups=subjects["Subject"]))
    X_train = X.iloc[train_idx].reset_index(drop=True)
    X_test = X.iloc[test_idx].reset_index(drop=True)
    y_train = y.iloc[train_idx].reset_index(drop=True)
    y_test = y.iloc[test_idx].reset_index(drop=True)

    return X_train, X_test, y_train, y_test

def to_binary_label(activity):
    if activity in ["WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS"]:
        return 1 # Active
    else:
        return 0 # Inactive
    
def make_unique(names):
    counts = {}
    unique_names = []
    for name in names:
        if name in counts:
            counts[name] += 1
            unique_names.append(f"{name}.{counts[name]}")
        else:
            counts[name] = 0
            unique_names.append(name)
    return unique_names


