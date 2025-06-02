import numpy as np

#gets all subsets of masks when one feature is fixed to 0.
#basically all c \in f/i 

def get_all_subsets(state_space_dim: int) -> list:
    """
    Generate all binary lists of given length with certain positions fixed to 0.
    
    Parameters
    ----------
    state_space_dim : int
        
    Returns
    -------
    variations : list
        list of all possible binary lists with the specified features fixed to 0
    """
    variations = []
    
    # calculate how many bits we need to vary (total length minus fixed positions)
    variable_positions = [pos for pos in range(state_space_dim)]
    num_variable_bits = len(variable_positions)
    
    # generate all possible combinations for the variable bits
    for num in range(2 ** num_variable_bits):
        binary = [0] * state_space_dim
        
        # fill in the variable positions
        for bit_pos in range(num_variable_bits):
            # get the current variable position in the original list
            original_pos = variable_positions[bit_pos]
            # get the bit value (0 or 1)
            bit_value = (num >> (num_variable_bits - 1 - bit_pos)) & 1
            binary[original_pos] = bit_value
            
        variations.append(binary)
    
    return variations

def get_all_group_subsets(G: list) -> np.ndarray:
    """
    Gets all permutations of subsets, with masked group fixed to 0.

    Parameters
    ----------
    g : list
    masked_group : int
        the group that is fixed to 0

    Returns
    -------
    permutations : np.ndarray
        all permutations of the groups, with masked group fixed to 0.   
    """

    # first index = group & second index = feature index
    n = sum(len(sublist) for sublist in G) # number of features
    k = len(G) # number of groups

    r = get_all_subsets(k)
    num_perms = len(r)

    permutations = np.ones((num_perms, n), dtype=np.int64)  # initialize the permutations array
    for l, rnoget in enumerate(r):
        p_i = np.ones(n)
        for i in range(k):
            # Loop over every j feature index in the i-th group.
            for j in G[i]: 
                p_i[j] = rnoget[i]
        permutations[l] = p_i
    return permutations
