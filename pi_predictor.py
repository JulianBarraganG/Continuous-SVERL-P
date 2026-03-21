import gymnasium as gym
from os.path import join

from sverl.shapley_utils import get_f_characteristic_dict
from sverl.globalvars import MODEL_FILEPATH, TRAJECTORY_SIZE, TRAJECTORY_FILENAME

def pi_pred_cartpole():

    env = gym.make("CartPole-v1")
    savepath=join("characteristic_dicts", "pi_predictor_cartpole.pkl")

    char_dict = get_f_characteristic_dict(savepath, env, MODEL_FILEPATH, TRAJECTORY_FILENAME, TRAJECTORY_SIZE,G=None) 
    
    return char_dict

if __name__ == "__main__":
    print("Running 'pi_predictor.py' directly. Purely for testing.")
    pi_pred_cartpole()
