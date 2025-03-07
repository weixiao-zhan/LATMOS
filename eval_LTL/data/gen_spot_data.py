import spot
import networkx as nx
import re
from itertools import product
from dataclasses import dataclass
import random
from tqdm import tqdm
import numpy as np
from sympy import symbols, sympify
import itertools
import pickle

@dataclass
class Automaton:
    graph: nx.classes.digraph.DiGraph
    init_state: int
    acc_states: set
    ap_dim: int

def label_to_bools(label, ap_dim):
    # Create symbolic variables p0, p1, ..., pn-1
    vars = symbols(f'p0:{ap_dim}')
    
    # Translate the expression to SymPy syntax
    label = label.replace('!', '~')
    
    # Convert the string expression to a SymPy expression
    try:
        expr = sympify(label, evaluate=False)
    except SyntaxError:
        print(f"Error: Could not parse the expression '{label}'. Please check the syntax.")
        return []
    
    # Generate all possible combinations of True and False for n variables
    all_combinations = itertools.product([1, 0], repeat=ap_dim)
    
    # Filter the combinations that satisfy the boolean expression
    satisfied_combinations = []
    for combo in all_combinations:
        # Create a dictionary to map each variable to its boolean value
        var_dict = dict(zip(vars, combo))
        
        # Substitute the values and evaluate the expression
        if expr.subs(var_dict):
            satisfied_combinations.append(combo)
    
    return satisfied_combinations


def spot_to_Automaton(aut, verbose=False):
    '''
    Convert spot to networkX-based Automaton 
    '''
    G = nx.DiGraph()
    ap_dim = len(aut.ap())
    bdict = aut.get_dict()

    # Add nodes (states)
    for s in range(aut.num_states()):
        G.add_node(s)

    init_state = aut.get_init_state_number()
    acc_states = set()

    # Add edges
    for s in range(aut.num_states()):
        for e in aut.out(s):
            t = e.dst if not aut.is_univ_dest(e.dst) else tuple(aut.univ_dests(e.dst))
            label = spot.bdd_format_formula(bdict, e.cond)
            G.add_edge(s, t, label=label_to_bools(label, ap_dim))
            if e.acc.count() >0: acc_states.add(s)

    # Set initial state attribute
    init = aut.get_init_state_number()
    G.graph['initial_state'] = init if not aut.is_univ_dest(init) else tuple(aut.univ_dests(init))

    if verbose:
        print(f"Number of nodes: {G.number_of_nodes()}")
        print(f"Number of edges: {G.number_of_edges()}")
        print(f"Initial state: {G.graph['initial_state']}")
        for u, v, data in G.edges(data=True):
            print(f"Edge {u} -> {v}:")
            print(f"  Label: {data['label']}")
        
        print("init state:", init_state)
        print("acc states:", acc_states)

    return Automaton(G, init_state, acc_states, ap_dim)

def check_ap_condition(condition, ap):
    """
    Check if the random AP satisfies the edge label condition.
    
    Args:
    condition (list): The edge label from the NetworkX graph, may contain None.
    ap (list): The randomly generated AP vector.
    
    Returns:
    bool: True if the random AP satisfies the edge label condition, False otherwise.
    """
    for condition_val, ap_val in zip(condition, ap):
        if condition_val is not None and condition_val != ap_val:
            return False
    return True

def state_to_one_hot(state, num_states):
    """
    Convert a state to its one-hot encoding.
    Args:
    state (int): The current state.
    num_states (int): The total number of states in the automaton.
    Returns:
    np.array: One-hot encoded state vector.
    """
    one_hot = np.zeros(num_states)
    one_hot[state] = 1
    return one_hot

def random_walk_automaton_once(A: Automaton, num_steps, 
                               ap_noise_var):
    current_state = A.init_state
    num_states = len(A.graph.nodes)
    state_vector = [current_state]
    ap_vector = []
    acceptance_vector = [current_state in A.acc_states]

    for _ in range(num_steps - 1):
        # valid_transitions = []
        # while not valid_transitions:
        #     # Generate a random boolean vector
        #     random_ap = tuple([random.choice([True, False]) for _ in range(A.ap_dim)])
        #     # Find valid transitions
        #     for neighbor in A.graph.neighbors(current_state):
        #         edge_data = A.graph.get_edge_data(current_state, neighbor)
        #         if random_ap in edge_data['label']:
        #             valid_transitions.append(neighbor)
        #     # If no valid transitions, we'll generate a new random_ap in the next iteration
        # next_state = random.choice(valid_transitions)

        next_state = random.choice(list(A.graph.successors(current_state)))
        random_ap = random.choice(A.graph.get_edge_data(current_state, next_state)['label'])
        if ap_noise_var != 0:
            noise = np.random.normal(0, np.sqrt(ap_noise_var), len(random_ap))
            random_ap = np.array(random_ap) + noise
        # Update vectors
        state_vector.append(next_state)
        ap_vector.append(random_ap)
        acceptance_vector.append(next_state in A.acc_states)

        # Move to the next state
        current_state = next_state

    return np.array(state_vector), np.array(ap_vector), np.array(acceptance_vector)

def random_walk_automaton(A: Automaton, num_steps, num_traces,
                          ap_noise_var):
    state_vectors, ap_vectors, acceptance_vectors = [], [], []
    for _ in tqdm(range(num_traces)):
        state_vector, ap_vector, acceptance_vector = random_walk_automaton_once(A, num_steps, ap_noise_var)
        state_vectors.append(state_vector)
        ap_vectors.append(ap_vector)
        acceptance_vectors.append(acceptance_vector)
    return np.array(state_vectors), np.array(ap_vectors), np.array(acceptance_vectors)

def gen_data_012():
    num_train_traces = 2**10  # Half the original traces for train
    num_val_traces = 2**8    # Half for validation
    TLTs = [
        '(p0 U p1) & GFp2 & GFp3',
        'Gp0 xor Xp1',
        'p1 xor (p2 xor (X(1 U p0) W p1))'
    ]
    automatons = []
    for idx, tlt in enumerate(TLTs):
        for train_var, val_var in zip(
                [0,   0,   0, 0.1, 0.2],
                [0, 0.1, 0.2,   0,   0]):
            aut = spot.translate(tlt, 'Buchi', 'state-based', 'unambiguous')
            A = spot_to_Automaton(aut)
            automatons.append(A)
            num_steps = len(A.graph.nodes)
            
            # Generate training data with noise
            train_state_vectors, train_ap_vectors, train_acceptance_vectors = random_walk_automaton(
                A, num_steps, num_train_traces,
                ap_noise_var=train_var
            )
            
            # Generate validation data without noise
            val_state_vectors, val_ap_vectors, val_acceptance_vectors = random_walk_automaton(
                A, num_steps, num_val_traces, 
                ap_noise_var=val_var
            )
            
            print(f"LTL {idx}:")
            print("Train - State vector:", train_state_vectors.shape)
            print("Train - AP vector:", train_ap_vectors.shape)
            print("Train - Acceptance vector:", train_acceptance_vectors.shape)
            print("Val - State vector:", val_state_vectors.shape)
            print("Val - AP vector:", val_ap_vectors.shape)
            print("Val - Acceptance vector:", val_acceptance_vectors.shape)
            
            np.savez(f'spot_{idx}_{train_var}_{val_var}',
                    train_state_vectors=train_state_vectors, 
                    train_ap_vectors=train_ap_vectors, 
                    train_acceptance_vectors=train_acceptance_vectors,
                    val_state_vectors=val_state_vectors,
                    val_ap_vectors=val_ap_vectors,
                    val_acceptance_vectors=val_acceptance_vectors)

def gen_data_3():
    num_train_traces = 2**10  # Half the original traces for train
    num_val_traces = 2**8    # Half for validation
    idx = 3

    aut = spot.translate('Gp0 xor Xp1', 'Buchi', 'state-based', 'unambiguous', 'complete')
    A_train = spot_to_Automaton(aut)
    A_train.graph.remove_edge(3, 1)

    A_val = spot_to_Automaton(aut)
    A_val.graph.remove_edge(3, 4)

    num_steps = len(A_train.graph.nodes)

    # Generate training data with noise
    train_state_vectors, train_ap_vectors, train_acceptance_vectors = random_walk_automaton(
        A_train, num_steps, num_train_traces,
        ap_noise_var=0
    )

    # Generate validation data without noise
    val_state_vectors, val_ap_vectors, val_acceptance_vectors = random_walk_automaton(
        A_val, num_steps, num_val_traces, 
        ap_noise_var=0
    )

    print(f"LTL {idx}:")
    print("Train - State vector:", train_state_vectors.shape)
    print("Train - AP vector:", train_ap_vectors.shape)
    print("Train - Acceptance vector:", train_acceptance_vectors.shape)
    print("Val - State vector:", val_state_vectors.shape)
    print("Val - AP vector:", val_ap_vectors.shape)
    print("Val - Acceptance vector:", val_acceptance_vectors.shape)

    np.savez(f'spot_{idx}_0_0',
        train_state_vectors=train_state_vectors, 
        train_ap_vectors=train_ap_vectors, 
        train_acceptance_vectors=train_acceptance_vectors,
        val_state_vectors=val_state_vectors,
        val_ap_vectors=val_ap_vectors,
        val_acceptance_vectors=val_acceptance_vectors)


# Change the main block to:
if __name__ == '__main__':
    spot.setup()
    gen_data_012()
    gen_data_3()