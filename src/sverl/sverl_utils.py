import numpy as np
from tqdm import trange
from gymnasium import Env

from .shapley_utils import shapley_value, global_sverl_value_function

def get_sverl_p(policy: callable,
                env: Env,
                feature_imputation_fnc: callable,
                G: list | None = None,
                num_rounds: int = 10,
                starting_state: np.ndarray | None = None,
                characteristic_fnc: callable = global_sverl_value_function) -> tuple[np.ndarray, float]:
    """
    Calculate the SVERL-P for the given policy and environment.
    groupShapley can be calculated, by specifying the groups in G.
    If G is None, then individual Shapley values are calculated.
    Parameters
    ----------
    policy : callable 
        $\\pi : \\mathcal{S} \\rightarrow \\mathcal{A}$
    env : Env
    feature_imputation_fnc : callable
    G : list | None
        Groups of features. If None, then individual Shapley values are calculated.
    num_rounds : int
    starting_state : np.ndarray | None
        For local SVERL-P.
    Returns
    -------
    shapley_values : numpy.ndarray
        SVERL-P values for each (group of) feature(s).
    value_empty_set : float

    TODO: Can't currently calculate local SVERL-P.
    """

    num_features = env.observation_space.shape[0]
    if G is None: # No grouping
        G = [[i] for i in range(num_features)]
    value_empty_set_sum = 0
    shapley_values_sum = np.zeros(num_features)
    for i in trange(num_rounds): 
        value_empty_set_sum += characteristic_fnc(policy, i, feature_imputation_fnc, np.zeros(num_features), env)
        for g in range(len(G)):
            shapley_values_sum[g] += shapley_value(policy, feature_imputation_fnc, characteristic_fnc, G, g, i, env)

    shapley_values = shapley_values_sum / num_rounds
    value_empty_set = value_empty_set_sum / num_rounds

    return shapley_values, value_empty_set

def report_sverl_p(shap_vls: np.ndarray,
                    state_feature_names: list[str]) -> None:
    """
    Report the Shapley values of the state features and the value of the empty set.

    Parameters
    ----------
    shap_vls : list
        Shapley values for each state feature.
    state_feature_names : list[str]
        Names of the state features.
    """
    right_adjust = max([(len(f_name)) for f_name in state_feature_names]) + 1

    for i, shap_vl in enumerate(shap_vls):
        print(f"Shapley value of {state_feature_names[i]:<{right_adjust}}: {shap_vl:>8.2f}")

