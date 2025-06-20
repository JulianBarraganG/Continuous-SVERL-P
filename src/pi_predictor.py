import gymnasium as gym
import numpy as np
from os.path import exists, join
import pickle as pkl

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from sverl.globalvars import CP_STATE_FEATURE_NAMES
from sverl.shapley_utils import get_f_characteristic_dict
from sverl.shapley_utils import shapley_value
from sverl.globalvars import MODEL_FILEPATH, TRAJECTORY_SIZE, TRAJECTORY_FILENAME

def pi_pred_cartpole():

    env = gym.make("CartPole-v1")
    savepath=join("characteristic_dicts", "pi_predictor_cartpole.pkl")

    char_dict = get_f_characteristic_dict(savepath, env, MODEL_FILEPATH, TRAJECTORY_FILENAME, TRAJECTORY_SIZE,G=None) 
    
    return char_dict

if __name__ == "__main__":
    print("Running 'pi_predictor.py' directly. Purely for testing.")
    pi_pred_cartpole()