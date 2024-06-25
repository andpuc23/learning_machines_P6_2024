#!/usr/bin/env python3
import sys
from robobo_interface import SimulationRobobo, HardwareRobobo
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
from task_1_model import Model, ReplayMemory, optimize_model, tau, select_action
import torch
import cv2
import numpy as np
import random


action_space = ['move_forward', 'move_backward', 'turn_left', 'turn_right']
food_threshold_val = 20

np.random.seed(0xC0FFEE)
torch.manual_seed(0xC0FFEE)
random.seed(0xC0FFEE)

def get_number_of_green_pixels(img) -> float:
    blue, green, red = cv2.split(img)
    px_num = blue.shape[0]*blue.shape[1]
    mask = (green > blue+10) & (green > red+10)
    count = np.count_nonzero(mask)
    return count / px_num


def get_number_of_red_pixels(img) -> float:
    blue, green, red = cv2.split(img)
    px_num = blue.shape[0]*blue.shape[1]
    mask = (red > blue+10) & (red > green+10)
    count = np.count_nonzero(mask)
    return count / px_num


def get_img_sectors_colors(img)->tuple:
    left = img[:img.shape[0]//3, :, :]
    center = img[img.shape[0]//3:img.shape[0]//3*2, :, :]
    right = img[img.shape[0]//3*2:, :, :]
    green_left = get_number_of_green_pixels(left)
    green_center = get_number_of_green_pixels(center)
    green_right = get_number_of_green_pixels(right)
    red_left = get_number_of_red_pixels(left)
    red_center = get_number_of_red_pixels(center)
    red_right = get_number_of_red_pixels(right)
    return green_left, green_center, green_right, red_left, red_center, red_right


def do_action(rob, action_idx, action_space):
    action = action_space[action_idx]
    if action == 'move_forward':
        rob.move_blocking(20, 20, 400)
    elif action == 'move_backward':
        rob.move_blocking(-20, -20, 400)
    elif action == 'turn_left':
        rob.move_blocking(-20, 20, 400)
    elif action == 'turn_right':
        rob.move_blocking(20, -20, 400)
    else:
        print('unknown action:', action_idx)


def get_observation(rob) -> list:
    """
    we give 4 irs sensors' values (front back, left right) 
    and divide the image in 3 sectors vertically 
    and count the number of green (target) and red (food) pixels in each
    """
    irs = rob.read_irs()
    irs = [irs[6], irs[4], irs[7], irs[5]] # backC, frontC, frontLL, frontRR
    img = rob.get_image_front()
    pixels = get_img_sectors_colors(img)
    return irs + list(pixels)



def _distance(rob, obj1, obj2) -> float:
    pos1 = rob._sim.getObjectPosition(obj1, rob._sim.handle_world)
    pos2 = rob._sim.getObjectPosition(obj2, rob._sim.handle_world)

    return np.sqrt((pos1[0]-pos2[0])**2 + (pos1[1] - pos2[1])**2)


def cost_function(rob):
    distance_rob_to_food = _distance(rob, rob, food)
    distance_food_to_base = _distance(rob, food, target)
    if distance_rob_to_food > food_threshold_val:
        return distance_rob_to_food + 10 # starting dist is 1.325...
    else:
        return distance_rob_to_food + distance_food_to_base


def train(rob):
    pass


def run(rob):
    q_table = np.zeros((1000, len(action_space)))  # Adjust the state space size as needed
    learning_rate = 0.1
    discount_factor = 0.9
    epsilon = 1.0
    epsilon_decay = 0.99
    min_epsilon = 0.1
    episodes = 1000

    for episode in range(episodes):
        state = get_observation(rob)
        state_idx = state_to_index(state)  # Function to convert state to index
        total_cost = 0
        
        for step in range(1000):  # limiting the number of steps per episode
            if random.uniform(0, 1) < epsilon:
                action = random.randint(0, len(action_space) - 1)
            else:
                action = np.argmin(q_table[state_idx, :])
            
            do_action(rob, action, action_space)
            next_state = get_observation(rob)
            next_state_idx = state_to_index(next_state)
            cost = cost_function(rob)
            
            # Update Q-values using cost function
            best_next_action = np.argmin(q_table[next_state_idx, :])
            td_target = cost + discount_factor * q_table[next_state_idx, best_next_action]
            q_table[state_idx, action] += learning_rate * (td_target - q_table[state_idx, action])
            
            state_idx = next_state_idx
            total_cost += cost
            
            if is_done(next_state):  # define your termination condition
                break
        
        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        print(f"Episode {episode + 1}/{episodes}, Total Cost: {total_cost}")

def state_to_index(state):
    # Convert the state to a unique index
    # This is a placeholder function; you need to implement a proper state indexing mechanism
    return int(sum(state)) % 1000

def is_done(state):
    # Define your termination condition based on the state
    # Placeholder condition:
    return False

if __name__ == "__main__":
    if sys.argv[1] == "--hardware":
        rob = HardwareRobobo(camera=True)
    elif sys.argv[1] == "--simulation":
        rob = SimulationRobobo()
        rob.play_simulation()
    else:
        raise ValueError(f"{sys.argv[1]} is not a valid argument, --simulation or --hardware expected.")
    
    
    food = rob._sim.getObject('/Food')
    target = rob._sim.getObject('/Base')
    
    if sys.argv[2] == '--train':
        training = True
    elif sys.argv[2] == '--test':
        training = False
    else:
        raise ValueError(f"{sys.argv[2]} is not a valid argument, --train or --test expected.")

    if training:
        train(rob)
    else:
        run(rob)
    
