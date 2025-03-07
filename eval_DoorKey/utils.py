import os
import numpy as np
import gymnasium as gym
import pickle
import matplotlib.pyplot as plt
import imageio
import random
from minigrid.core.world_object import Goal, Key, Door

ST = 0  # Stay
MF = 1  # Move Forward
TL = 2  # Turn Left
TR = 3  # Turn Right
PK = 4  # Pickup Key
UD = 5  # Unlock Door

control_str = {
    ST: "ST",
    MF: "MF",
    TL: "TL",
    TR: "TR",
    PK: "PK",
    UD: "UD"
}


def step_cost(action):
    # You should implement the stage cost by yourself
    # Feel free to use it or not
    # ************************************************
    if action == ST:
        return 0
    elif action == MF:
        return 2
    else:
        return 1

def step(env, action):
    """
    Take Action
    ----------------------------------
    actions:
        0 # Move forward (MF)
        1 # Turn left (TL)
        2 # Turn right (TR)
        3 # Pickup the key (PK)
        4 # Unlock the door (UD)
    """
    actions = {
        1: env.unwrapped.actions.forward,
        2: env.unwrapped.actions.left,
        3: env.unwrapped.actions.right,
        4: env.unwrapped.actions.pickup,
        5: env.unwrapped.actions.toggle,
    }

    (obs, reward, terminated, truncated, info) = env.step(actions[action])
    return step_cost(action), terminated


def generate_random_env(seed, task):
    """
    Generate a random environment for testing
    -----------------------------------------
    seed:
        A Positive Integer,
        the same seed always produces the same environment
    task:
        'MiniGrid-DoorKey-5x5-v0'
        'MiniGrid-DoorKey-6x6-v0'
        'MiniGrid-DoorKey-8x8-v0'
    """
    if seed < 0:
        seed = np.random.randint(50)
    env = gym.make(task, render_mode="rgb_array")
    env.reset(seed=seed)
    return env


def load_env(path):
    """
    Load Environments
    ---------------------------------------------
    Returns:
        gym-environment, info
    """
    with open(path, "rb") as f:
        env = pickle.load(f)

    info = {
        "height": env.unwrapped.height, 
        "width": env.unwrapped.width, 
        "init_agent_pos": env.unwrapped.agent_pos, 
        "init_agent_dir": env.unwrapped.dir_vec,
    }

    for i in range(env.unwrapped.height):
        for j in range(env.unwrapped.width):
            if isinstance(env.unwrapped.grid.get(j, i), Key):
                info["key_pos"] = np.array([j, i])
            elif isinstance(env.unwrapped.grid.get(j, i), Door):
                info["door_pos"] = np.array([j, i])
            elif isinstance(env.unwrapped.grid.get(j, i), Goal):
                info["goal_pos"] = np.array([j, i])

    return env, info


def load_random_env(env_folder):
    """
    Load a random DoorKey environment
    ---------------------------------------------
    Returns:
        gym-environment, info, path
    """
    env_list = [os.path.join(env_folder, env_file) for env_file in os.listdir(env_folder) if env_file.endswith('.env')]
    env_path = random.choice(env_list)
    with open(env_path, "rb") as f:
        env = pickle.load(f)

    info = {
        "height": env.unwrapped.height,
        "width": env.unwrapped.width,
        "init_agent_pos": env.unwrapped.agent_pos,
        "init_agent_dir": env.unwrapped.dir_vec,
        "door_pos": [],
        "door_open": [],
    }

    for i in range(env.unwrapped.height):
        for j in range(env.unwrapped.width):
            if isinstance(env.get_wrapper_attr('grid').get(j, i), Key):
                info["key_pos"] = np.array([j, i])
            elif isinstance(env.get_wrapper_attr('grid').get(j, i), Door):
                info["door_pos"].append(np.array([j, i]))
                if env.get_wrapper_attr('grid').get(j, i).is_open:
                    info["door_open"].append(True)
                else:
                    info["door_open"].append(False)
            elif isinstance(env.get_wrapper_attr('grid').get(j, i), Goal):
                info["goal_pos"] = np.array([j, i])

    return env, info, env_path

def load_all_random_env(env_folder):
    """
    Load a random DoorKey environment
    ---------------------------------------------
    Returns:
        gym-environment, info, path
    """
    env_list = [os.path.join(env_folder, env_file) for env_file in os.listdir(env_folder) if env_file.endswith('.env')]
    re = []
    for env_path in env_list:
        with open(env_path, "rb") as f:
            env = pickle.load(f)
    
        info = {
            "height": env.unwrapped.height,
            "width": env.unwrapped.width,
            "init_agent_pos": env.unwrapped.agent_pos,
            "init_agent_dir": env.unwrapped.dir_vec,
            "door_pos": [],
            "door_open": [],
        }

        for i in range(env.unwrapped.height):
            for j in range(env.unwrapped.width):
                if isinstance(env.get_wrapper_attr('grid').get(j, i), Key):
                    info["key_pos"] = np.array([j, i])
                elif isinstance(env.get_wrapper_attr('grid').get(j, i), Door):
                    info["door_pos"].append(np.array([j, i]))
                    if env.get_wrapper_attr('grid').get(j, i).is_open:
                        info["door_open"].append(True)
                    else:
                        info["door_open"].append(False)
                elif isinstance(env.get_wrapper_attr('grid').get(j, i), Goal):
                    info["goal_pos"] = np.array([j, i])

        re.append((env, info, env_path))
    return re

def load_one_random_env(env_path):
    with open(env_path, "rb") as f:
        env = pickle.load(f)

    info = {
        "height": env.unwrapped.height,
        "width": env.unwrapped.width,
        "init_agent_pos": env.unwrapped.agent_pos,
        "init_agent_dir": env.unwrapped.dir_vec,
        "door_pos": [],
        "door_open": [],
    }

    for i in range(env.unwrapped.height):
        for j in range(env.unwrapped.width):
            if isinstance(env.get_wrapper_attr('grid').get(j, i), Key):
                info["key_pos"] = np.array([j, i])
            elif isinstance(env.get_wrapper_attr('grid').get(j, i), Door):
                info["door_pos"].append(np.array([j, i]))
                if env.get_wrapper_attr('grid').get(j, i).is_open:
                    info["door_open"].append(True)
                else:
                    info["door_open"].append(False)
            elif isinstance(env.get_wrapper_attr('grid').get(j, i), Goal):
                info["goal_pos"] = np.array([j, i])

    return (env, info, env_path)

def save_env(env, path):
    with open(path, "wb") as f:
        pickle.dump(env, f)


def plot_env(env, path = None):
    """
    Plot current environment
    ----------------------------------
    """
    img = env.render()
    plt.figure()
    plt.imshow(img)
    if path is not None:
        plt.savefig(path)
    plt.show()

def draw_gif_from_seq(seq, env, path="./gif/doorkey.gif"):
    """
    Save gif with a given action sequence
    ----------------------------------------
    seq:
        Action sequence, e.g [0,0,0,0] or [MF, MF, MF, MF]

    env:
        The doorkey environment
    """
    with imageio.get_writer(path, mode="I", duration=0.8) as writer:
        img = env.render()
        writer.append_data(img)
        for act in seq:
            step(env, act)
            img = env.render()
            writer.append_data(img)
    print(f"GIF is written to {path}")
    return
    
    
