import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from library import *
from scipy.integrate import solve_ivp
import pickle
import os
import sympy as sp
from tqdm import tqdm
from pprint import pprint
from matplotlib.patches import Ellipse
from matplotlib.patches import Circle

import gym
import numpy as np
from gym import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import random
import torch


# Load Map Data
X, Y, map_image, obstacleMap, xLM, yLM, Xvec, Yvec = load_data()

# Define Map Resolution
mapRes = 10     # meters per pixel

# Define Physical Values
L = 3           # m - axles distance
v_max = 4e-2    # m/s - max speed

# Define Initial and Intermediate Poses
P0 = np.array([42.38*1e3, 11.59*1e3, np.pi/2])          # m, m, rad
P1 = np.array([33.07*1e3, 19.01*1e3, np.pi])            # m, m, rad

# P0 = {'rho': 42.38*1e3, 'alpha': 11.59*1e3, 'beta': np.pi/2}
# P1 = {'rho': 33.07*1e3, 'alpha': 19.01*1e3, 'beta': np.pi}

figres = 300

# Define Viapoints
viapoints = np.array([P1])      # free trajectory

# Select Tolerance
rho_tol = 0.1  # m

# Initialize Variables
P = np.empty((3, 0))  # P is 3 x N
u = np.empty((2, 0))  # u is 2 x N
tspan = np.array([])
dt = 1


class GaleCrater(gym.Env):
    def __init__(self, P_start, P_end, dt):
        super(GaleCrater, self).__init__()

        self.t_start = 0
        self.dt = dt
        self.P_start = P_start
        self.P_end = P_end
        self.R_start = P2R(P_start)
        self.R_end = P2R(P_end)

        # Define the contonuous action space for control gains
        self.action_space = spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)
        
        # Define the observation space as the current state of the system
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)

        # Initialize State and Target State
        self.state = self.R_start
        self.target_state = self.R_end


    def reset(self, **kwargs):
        '''
        Reset the environment to an initial state.
        '''

        self.state = self.R_start
        self.state_buffer = []
        self.step_id = 0
        infos = {}                  # neglet reset infos needed from Gymnasium

        return self.flatten(self.state), infos
        

    def step(self, action):

        # Retrieve Gains and State
        Krho, Kalpha, Kbeta = action
        rho, alpha, beta = self.state

        # Compute Control
        v = Krho * rho
        omega = Kalpha * alpha + Kbeta * beta

        # Set Control Constraints
        v_max = 4e-2    # m/s - max speed
        
        if v > v_max:
            v = v_max

        # Compute Derivatives
        drho = -np.cos(alpha) * v
        dalpha = np.sin(alpha) * v / rho - omega
        dbeta = -np.sin(alpha) * v / rho

        # Compute New State
        rho_new = rho + drho * self.dt
        alpha_new = alpha + dalpha * self.dt
        beta_new = beta + dbeta * self.dt

        # Update State
        self.state = np.array([rho_new, alpha_new, beta_new])

        # Compute reward as distance from target
        reward = -np.linalg.norm(self.state[:2] - self.target_state[:2])

        # Check if done
        done = np.linalg.norm(self.state[:2] - self.target_state[:2]) < 0.1

        # Optional: add any additional info
        info = {}

        return self.state, reward, done, info



# Perform the Integration
for i in range(len(viapoints)):

    P_int, u_int, tspan_int = KinematicModel(P_start, viapoints[i], K, rho_tol, t_start)

    # Stack the results
    P = np.hstack((P, P_int))
    u = np.hstack((u, u_int))
    tspan = np.concatenate((tspan, tspan_int))

    # Check for obstacles
    P_indices = np.array([get_indices(P_int[0:2, i], X, Y, mapRes) for i in range(P_int.shape[1])])
    if check_obstacles(P_indices, obstacleMap):
        print('\n\nThe Trajectory intersects with an Obstacle!\n')
        break

    P_start = P_int[:, -1]          # store the real final state -> start of next segment
    t_start = tspan_int[-1]         # store the real final time -> start of next segment


P1r = P[:, -1]  # store the real final state

# Decide which device we want to run on
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

if __name__ == '__main__':

    max_m_episode = 800000
    max_steps = 800

    env = GaleCrater(P0, P1r, dt)

    # Load the model or create a new one
    model_path = os.path.join('Models', 'model_ppo')
    if os.path.exists(model_path + ".zip"):
        model = PPO.load(model_path, env=env)  # Set the environment here
    else:
        model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.3)

    # Train the model
    model.learn(total_timesteps=1000000, progress_bar=True)  # Adjust the number of timesteps as needed

    # Save the model
    model_path = os.path.join('models', 'model_ppo')
    model.save(model_path)
