from utils import *
from tqdm import tqdm
from typing import Any, Generator, Tuple
import numpy as np
import torch
import pickle
import matplotlib.pyplot as plt
import gymnasium as gym
from minigrid.envs.doorkey import DoorKeyEnv
from minigrid.core.grid import Grid
from minigrid.core.world_object import Wall, Goal, Key, Door
from functools import total_ordering


class XY:
    def __init__(self, x = 0, y = 0):
        self.x = x
        self.y = y
    
    def init(self, x, y):
        self.x = x
        self.y = y
        return self
    
    def init_from(self, other):
        self.x = other.x
        self.y = other.y
        return self
    
    def __add__(self, other):
        if not isinstance(other, XY):
            return NotImplemented
        return XY(self.x+other.x, self.y+other.y)
    
    def __mul__(self, dir):
        if dir == TL:
            return XY(self.y, - self.x)
        elif dir == TR:
            return XY(- self.y, self.x)
        else:
            raise ValueError() 
        
    def __eq__(self, other: "XY"):
        return self.x == other.x and self.y == other.y

    def __repr__(self):
        return (f"({self.x}, {self.y})")
    
    def __hash__(self):
        return hash((self.x, self.y))

@total_ordering
class State:
    def __init__(self, door_num): 
        """
        n: num of doors
        """
        self.agent_pos = XY()
        self.agent_dir = XY()
        self.goal_pos  = XY()
        
        self.key_pos   = XY()
        self.key_carrying = False

        self.door_num = door_num
        self.door_pos  = tuple([XY() for _ in range(door_num)])
        self.door_is_open = tuple([False for _ in range(door_num)])

    def init(self, agent_pos: XY, agent_dir: XY, goal_pos: XY, 
             key_pos: XY, key_carrying: bool, 
             door_pos: Tuple[XY, ...], door_is_open: Tuple[bool, ...]):
        self.agent_pos.init_from(agent_pos)
        self.agent_dir.init_from(agent_dir)
        self.goal_pos.init_from(goal_pos)

        self.key_pos.init_from(key_pos)
        self.key_carrying = key_carrying
        
        for i in range(self.door_num):
            self.door_pos[i].init_from(door_pos[i])
        self.door_is_open = door_is_open
        return self

    def init_from_state(self, other: "State"):
        self.agent_pos.init_from(other.agent_pos)
        self.agent_dir.init_from(other.agent_dir)
        self.goal_pos.init_from(other.goal_pos)
        self.key_pos.init_from(other.key_pos)
        self.key_carrying = other.key_carrying
        for i in range(self.door_num):
            self.door_pos[i].init_from(other.door_pos[i])
        self.door_is_open = other.door_is_open
        return self

    def __hash__(self):
            return hash((self.agent_pos, self.agent_dir, 
                         self.goal_pos, 
                         self.key_pos, self.key_carrying, 
                         self.door_pos, self.door_is_open))

    def __eq__(self, other):
        if not isinstance(other, State):
            return NotImplemented
        return self.__hash__() == other.__hash__()
    
    def __lt__(self, other):
        return self.__hash__() > other.__hash__()
    
    def __repr__(self) -> str:
        return f"State:{self.__hash__()}\n\tAgent:{self.agent_pos}, {self.agent_dir}\n\tGoal:{self.goal_pos}\n\tKey:{self.key_pos}, {self.key_carrying}\n\tDoor:{self.door_pos},{self.door_is_open}\n"

    def to_vec(self):
        vec = [
            self.agent_pos.x, self.agent_pos.y, 
            self.agent_dir.x, self.agent_dir.y, 
            self.goal_pos.x, self.goal_pos.y, 
            self.key_pos.x, self.key_pos.y, 
            self.key_carrying
        ]
        for door_idx in range(self.door_num):
            vec += [self.door_pos[door_idx].x, self.door_pos[door_idx].y, self.door_is_open[door_idx]]
        return torch.tensor(vec)
    
    def from_vec(self, vec):
        self.agent_pos.x = int(vec[0])
        self.agent_pos.y = int(vec[1])
        self.agent_dir.x = int(vec[2])
        self.agent_dir.y = int(vec[3])
        self.goal_pos.x = int(vec[4])
        self.goal_pos.y = int(vec[5])
        self.key_pos.x = int(vec[6])
        self.key_pos.y = int(vec[7])
        self.key_carrying = bool(vec[8])
        self.door_num = (len(vec) - 9) // 3
        door_is_open = [None for _ in range(self.door_num)]
        for i in range(self.door_num):
            self.door_pos[i].x = int(vec[9 + 3*i])
            self.door_pos[i].y = int(vec[10 + 3*i])
            door_is_open[i] = bool(vec[11 + 3*i])
        self.door_is_open = tuple(door_is_open)
        return self

    def render(self):
        """
        Render the current state of the environment using Minigrid
        """
        env_wrapper = gym.make(f"MiniGrid-DoorKey-8x8-v0", render_mode="rgb_array")
        env_wrapper.reset()
        env = env_wrapper.env.env
        env.grid = Grid(8, 8)
        env.grid.vert_wall(4, 0)

        have_closed_door = False
        for i in range(self.door_num):
            if self.door_is_open[i]:
                door = Door(color="yellow", is_open=True)
            else:
                have_closed_door = True
                door = Door(color="yellow", is_open=False, is_locked=True)
            env.grid.set(self.door_pos[i].x, self.door_pos[i].y, door)
        if not self.key_carrying and have_closed_door:
            env.grid.set(self.key_pos.x, self.key_pos.y, Key(color="yellow"))
        env.grid.set(self.goal_pos.x, self.goal_pos.y, Goal())
        env.agent_pos = (self.agent_pos.x, self.agent_pos.y)
        

        env.agent_dir = \
            1 if self.agent_dir == XY(0,1) else \
            0 if self.agent_dir == XY(1,0) else \
            3 if self.agent_dir == XY(0,-1) else \
            2

        env.gen_obs()
        image = env.render()
        return image

    def abs_render(self):
        """
        Render the current state as a one-hot encoded tensor.
        Returns a tensor of shape (8, 8, num_channels) where num_channels includes:
        - Wall (1 channel)
        - Door (1 channel)
        - Open Door (1 channel)
        - Key (1 channel)
        - Goal (1 channel)
        - Agent facing right (1 channel)
        - Agent facing down (1 channel)
        - Agent facing left (1 channel)
        - Agent facing up (1 channel)
        Total channels: 9
        """
        # Initialize empty grid
        grid = torch.zeros((8, 8, 9), dtype=torch.float32)
        
        # Set wall (channel 0)
        grid[0:8, 4, 0] = 1  # Vertical wall
        
        # Set doors (channels 1 and 2)
        for i in range(self.door_num):
            x, y = self.door_pos[i].x, self.door_pos[i].y
            if self.door_is_open[i]:
                grid[y, x, 2] = 1  # Open door
            else:
                grid[y, x, 1] = 1  # Closed door
        
        # Set key (channel 3)
        if not self.key_carrying:
            grid[self.key_pos.y, self.key_pos.x, 3] = 1
        
        # Set goal (channel 4)
        grid[self.goal_pos.y, self.goal_pos.x, 4] = 1
        
        # Set agent position and direction (channels 5-8)
        agent_x, agent_y = self.agent_pos.x, self.agent_pos.y
        if self.agent_dir == XY(1, 0):      # Facing right
            grid[agent_y, agent_x, 5] = 1
        elif self.agent_dir == XY(0, 1):     # Facing down
            grid[agent_y, agent_x, 6] = 1
        elif self.agent_dir == XY(-1, 0):    # Facing left
            grid[agent_y, agent_x, 7] = 1
        elif self.agent_dir == XY(0, -1):    # Facing up
            grid[agent_y, agent_x, 8] = 1    
        return grid
    
class DoorKeySolver:
    def __init__(self):
        pass

    def is_wall(self, pos: XY) -> bool:
        raise NotImplementedError

    def motion_model(self, state: State, control):
        new_state = State(state.door_num).init_from_state(state)
        
        if control == ST: # stay
            if state.agent_pos == state.goal_pos:
                return new_state, 0
            else:
                return None, float("inf")
        
        if control == TL: # Turn Left
            new_state.agent_dir = state.agent_dir * TL
            return new_state, 1
        
        if control == TR: # Turn Right
            new_state.agent_dir = state.agent_dir * TR
            return new_state, 1

        front_pos = state.agent_pos + state.agent_dir

        if control == MF: # Move Forward
            if self.is_wall(front_pos) or \
                any(front_pos == state.door_pos[i] and (not state.door_is_open[i]) 
                    for i in range(state.door_num)):
                return None, float("inf")
            new_state.agent_pos = front_pos
            return new_state, 2

        if control == PK: # Pickup Key
            if (front_pos != state.key_pos):
                return None, float("inf")
            new_state.key_carrying = True
            return new_state, 1

        if control == UD: # Unlock Door
            for i in range(state.door_num):
                if front_pos == state.door_pos[i] and \
                        (not state.door_is_open[i]) and \
                        state.key_carrying:
                    tmp_list = list(state.door_is_open)
                    tmp_list[i] = True
                    new_state.door_is_open = tuple(tmp_list)
                    new_state.key_carrying = False
                    return new_state, 1
            return None, float("inf")
    
        raise ValueError("not supported control")
    
    def terminate_cost(self, state:State):
        if state.agent_pos == state.goal_pos:
            return 0
        return float("inf")
    
    def iter_state_space(self)-> Generator[State, Any, None]:
        raise NotImplementedError
    
    def iter_control_space(self):
        for u in [ST,MF,TL,TR,PK,UD]:
            yield u

    def solve(self, time_horizon):
        self.time_horizon = time_horizon
        print(f"solving with time_horizon={self.time_horizon}")
        self.V_t = dict()
        self.Pi_t = dict()

        V = dict()
        for state in self.iter_state_space():
            terminate_cost = self.terminate_cost(state)
            if terminate_cost < float('inf'):
                V[state] = terminate_cost
        self.V_t[self.time_horizon] = V

        for t in tqdm(range(self.time_horizon-1,-1,-1)):
            V_next = self.V_t[t + 1]  # Get the value function of the next time step
            V = dict()
            Pi = dict()
            for state in self.iter_state_space():
                best_cost = float('inf')
                best_control = None
                for control in self.iter_control_space():
                    new_state, stage_cost = self.motion_model(state, control)
                    if new_state is not None:
                        cost = stage_cost + V_next.get(new_state, float("inf"))
                    else:
                        cost = float("inf")
                    if cost < best_cost:
                        best_cost = cost
                        best_control = control
                if best_control is not None:
                    V[state] = best_cost
                    Pi[state] = best_control
            
            if V == V_next:
                print(f"DP early stopped at {t}")
                break

            self.V_t[t] = V
            self.Pi_t[t] = Pi
    
    def query(self, state: State, verbose = False):
        t0 = min(self.V_t.keys())
        total_cost = self.V_t[t0][state]
        controls = []
        states = [state]
        for t in range(t0, self.time_horizon):
            best_control = self.Pi_t[t][state]
            if best_control == ST:
                break
            controls.append(best_control)

            if state is None:
                raise TypeError("should never happen")
            state, _ = self.motion_model(state, best_control)
            states.append(state)
        if verbose: print(f"result: optimal cost: {total_cost}\noptimal control policy: {[control_str[control] for control in controls]}")
        return total_cost, controls, states

    def debug_first_time_sourced(self, state_hash):
        for t in range(self.time_horizon, -1, -1):
            for ss in self.V_t[t].keys():
                if ss.__hash__() == state_hash:
                    return t, self.V_t[t][ss]
        return -1,-1

class DoorKeySolver_1(DoorKeySolver):
    def __init__(self, env, info) -> None:
        DoorKeySolver().__init__()
        self.env, self.info = env, info
    
    def is_wall(self, pos):
        return isinstance(self.env.get_wrapper_attr('grid').get(pos.x, pos.y), Wall)

    def iter_state_space(self):
        key_pos = XY(*self.info["key_pos"])
        door_pos = XY(*self.info["door_pos"])
        goal_pos = XY(*self.info["goal_pos"])
        for x in range(self.info["height"]):
            for y in range(self.info["width"]):
                agent_pos = XY(x, y)
                if self.is_wall(agent_pos):
                    continue
                for agent_dir in [XY(0,1), XY(1,0),XY(0,-1),XY(-1,0)]:
                    for key_carrying in [True, False]:
                        for door_open in [True, False]:    
                            yield State(1).init(
                                agent_pos, agent_dir, goal_pos, 
                                key_pos, key_carrying, 
                                (door_pos,), (door_open,))

    def gen_init_state(self):
        return State(1).init(
            XY(*self.info["init_agent_pos"]),
            XY(*self.info["init_agent_dir"]),
            XY(*self.info["goal_pos"]),
            XY(*self.info["key_pos"]), False,
            (XY(*self.info["door_pos"]),),
            (self.env.get_wrapper_attr('grid').get(self.info["door_pos"][0], self.info["door_pos"][1]).is_open,)
        )
    
    def solve(self):
        time_horizon = len(list(self.iter_state_space()))
        return super().solve(time_horizon)

class DoorKeySolver_2(DoorKeySolver):
    def __init__(self) -> None:
        DoorKeySolver().__init__()
    
    def is_wall(self, pos: XY):
        if pos.x < 0 or pos.x > 7 or pos.y < 0 or pos.y > 7 or \
            (pos.x == 4 and (pos.y != 2 and pos.y != 5)):
            return True
        return False

    def iter_state_space(self):
        door_num = 2
        door_pos = (XY(4,2), XY(4,5))

        for x in range(8):
            for y in range(8):
                agent_pos = XY(x, y)
                if self.is_wall(agent_pos):
                    continue
                for agent_dir in [XY(0,1), XY(1,0), XY(0,-1), XY(-1,0)]:
                    for goal_pos in [XY(5,1), XY(6,3), XY(5,6)]:
                        for key_pos in [XY(1,1), XY(2,3), XY(1,6)]:
                            for key_carrying in [True, False]:
                                for door_open in [(True, True), (True, False), (False, True), (False, False)]:
                                    yield State(door_num).init(
                                        agent_pos, agent_dir, goal_pos, 
                                        key_pos, key_carrying, 
                                        door_pos, door_open)

    def gen_init_state(self, info):
        return State(2).init(
            XY(*info["init_agent_pos"]),
            XY(*info["init_agent_dir"]),
            XY(*info["goal_pos"]),
            XY(*info["key_pos"]),
            False,
            (
                XY(*info["door_pos"][0]),
                XY(*info["door_pos"][1])
            ),
            tuple(info["door_open"])
        )
    
    def solve(self, cashed_result_pickle = "DP_result.pkl"):
        time_horizon = len(list(self.iter_state_space()))

        if cashed_result_pickle is not None and \
        os.path.exists(cashed_result_pickle):
            self.time_horizon = time_horizon
            self.V_t, self.Pi_t = pickle.load(open(cashed_result_pickle, "rb"))
            return
        
        super().solve(time_horizon)
        if cashed_result_pickle is not None:
            pickle.dump((self.V_t, self.Pi_t), open(cashed_result_pickle, "wb"))