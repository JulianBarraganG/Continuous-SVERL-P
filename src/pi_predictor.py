import gymnasium as gym
import numpy as np
from os.path import exists, join
import pickle as pkl

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from sverl.group_utils import get_all_subsets
from sverl.globalvars import CP_STATE_FEATURE_NAMES
from utils import get_supervised_learning_data

env = gym.make("CartPole-v1")
state_space_dim = env.observation_space.shape[0] # type: ignore
for C in get_all_subsets(state_space_dim):

    if sum(C) == 0:
        print("Skipping empty feature set. Implement later.")
        continue

    feature_names = [CP_STATE_FEATURE_NAMES[i] for i in range(len(C)) if C[i] == 1]
    print(f"Processing state features:")
    for feature in feature_names:
        print(f"  - {feature}")
    print("Loading the data...")
    X_train, y_train, X_test, y_test = get_supervised_learning_data(C, env)

    # Fit the scaler
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    log = LogisticRegression()
    print(f"Fitting logistic regression model for {feature_names}...")
    log.fit(X_train, y_train)
    print(f"Evaluating logistic regression model for {feature_names}...")
    test_transformed = scaler.transform(X_test)
    print(f"Train score: {log.score(X_train, y_train):.2f}")
    print(f"Test score: {log.score(test_transformed, y_test):.2f}")

