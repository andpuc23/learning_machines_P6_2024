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
food_threshold_val = 0.02
rounding_value = 0.05

np.random.seed(0xC0FFEE)
torch.manual_seed(0xC0FFEE)
random.seed(0xC0FFEE)

def get_number_of_green_pixels(img) -> float:
    blue, green, red = cv2.split(img)
    px_num = blue.shape[0]*blue.shape[1]
    mask = (green > blue+10) & (green > red+10)
    count = np.count_nonzero(mask)
    return count / px_num * 100

def get_number_of_red_pixels(img) -> float:
    blue, green, red = cv2.split(img)
    px_num = blue.shape[0]*blue.shape[1]
    mask = (red > blue+10) & (red > green+10)
    count = np.count_nonzero(mask)
    return count / px_num * 100

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
    cv2.imwrite('/root/results/img.png', img)
    pixels = get_img_sectors_colors(img)
    # print('get_observation', 'red:', pixels[3:])
    return irs + list(pixels)

def _distance_rob_to_obj(rob, obj) -> float:
    rob_pos = rob.get_position()
    rob_pos = [rob_pos.x, rob_pos.y]
    obj_pos = rob._sim.getObjectPosition(obj, rob._sim.handle_world)
    return np.sqrt((rob_pos[0]-obj_pos[0])**2 + (rob_pos[1] - obj_pos[1])**2)

def _distance(rob, obj1, obj2) -> float:
    pos1 = rob._sim.getObjectPosition(obj1, rob._sim.handle_world)
    pos2 = rob._sim.getObjectPosition(obj2, rob._sim.handle_world)
    return np.sqrt((pos1[0]-pos2[0])**2 + (pos1[1] - pos2[1])**2)

def cost_function(rob, state):
    distance_rob_to_food = _distance_rob_to_obj(rob, food)
    distance_food_to_base = _distance(rob, food, target)
    cost = distance_rob_to_food
    cost += sum(state[:4])*0.01
    if distance_rob_to_food > food_threshold_val:
        return cost + 10 # starting dist is 1.325...
    else:
        return cost + distance_food_to_base

def train(rob):
    episode_length = 300
    learning_rate = 0.1
    discount_factor = 0.9
    epsilon = 1.0
    epsilon_decay = 0.99
    min_epsilon = 0.1
    episodes_num = 10
    q_table_size = 10000
    state_shape = 10

    q_table = np.zeros((q_table_size, len(action_space)))  # Adjust the state space size as needed
    rounded_states_list = np.zeros((q_table_size, state_shape))

    for episode in range(episodes_num):
        state = get_observation(rob)
        state_idx = state_to_index(state, rounded_states_list)  # Function to convert state to index
        total_cost = 0
        
        for _ in range(episode_length):
            if random.uniform(0, 1) < epsilon:
                action = random.randint(0, len(action_space) - 1)
            else:
                action = np.argmin(q_table[state_idx, :])
            
            do_action(rob, action, action_space)
            next_state = get_observation(rob)
            next_state_idx = state_to_index(next_state, rounded_states_list)
            cost = cost_function(rob, next_state)
            
            # Update Q-values using cost function
            best_next_action = np.argmin(q_table[next_state_idx, :])
            td_target = cost + discount_factor * q_table[next_state_idx, best_next_action]
            q_table[state_idx, action] += learning_rate * (td_target - q_table[state_idx, action])
            
            state_idx = next_state_idx
            total_cost += cost
            
            if is_done(next_state):
                break
        
        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        print(f"Episode {episode + 1}/{episodes_num}, Total Cost: {total_cost}")

    np.savez('/root/results/qtable', q_table)

def run(rob):
    q_table = np.load('/root/results/qtable.npz')['arr_0']
    q_table_size = 1000
    state_shape = 10
    rounded_states_list = np.zeros((q_table_size, state_shape))

    while True:
        state = get_observation(rob)
        state_idx = state_to_index(state, rounded_states_list)
        action = np.argmin(q_table[state_idx, :])
        do_action(rob, action, action_space)

def _find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    diff = np.abs(array - value)[idx].sum()
    return idx, diff

def state_to_index(state, states_list):
    global rounding_value
    rounded_obs = np.array([round(s / rounding_value) * rounding_value for s in state])
    rounded_obs = np.clip(rounded_obs[:4], 0, 150).tolist() + rounded_obs[4:].tolist()
    print('rounded:', rounded_obs)
    index, delta = _find_nearest(states_list, rounded_obs)

    if np.all(states_list[index] == 0) and np.all(rounded_obs != 0):
        states_list[index] = rounded_obs
        return index
    
    if delta > rounding_value * 3:
        index = (states_list == 0).all(axis=1).argmax()
        states_list[index] = rounded_obs
    
    return index

def is_done(state):
    global food_threshold_val
    return _distance_rob_to_obj(rob, food) < food_threshold_val

if __name__ == "__main__":
    if sys.argv[1] == "--hardware":
        rob = HardwareRobobo(camera=True)
    elif sys.argv[1] == "--simulation":
        rob = SimulationRobobo()
        rob.play_simulation()
        rob.set_phone_tilt(110, 50)
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
