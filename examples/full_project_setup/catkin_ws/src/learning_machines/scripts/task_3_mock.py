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
    green_left = get_number_of_green_pixels(left) * 10
    green_center = get_number_of_green_pixels(center) * 10
    green_right = get_number_of_green_pixels(right) * 10
    red_left = get_number_of_red_pixels(left) * 10
    red_center = get_number_of_red_pixels(center) * 10
    red_right = get_number_of_red_pixels(right) * 10
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
    total 10 numbers
    """
    irs = rob.read_irs()
    irs = [irs[6], irs[4], irs[7], irs[5]] # backC, frontC, frontLL, frontRR
    img = rob.get_image_front()
    pixels = get_img_sectors_colors(img)
    return irs + list(pixels)



def _distance_to_food(rob) -> float:
    rob_position = [rob.get_position().x, rob.get_position().y]
    return np.sqrt((rob_position[0] + 3.65)**2 + (rob_position[1] - 0.825)**2)


def _find_nearest(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        diff = np.abs(array - value)[idx].sum()
        return idx, diff

def cost_function(rob):
    return _distance_to_food(rob)


def state_to_index(state, states_list):
    rounding_value = 0.05
    index_to_insert_into = np.argwhere(states_list == np.zeros((10)))
    
    rounded_obs = np.array([round(s / rounding_value) * rounding_value for s in state])
    index, delta = _find_nearest(states_list, rounded_obs)

    if index_to_insert_into.shape[0] != 0 and delta > rounding_value*3:
        states_list[index_to_insert_into, :] = rounded_obs
        return index_to_insert_into
    
    return index
        


def is_done(state):
    food_threshold_val = 20
    return _distance_to_food(rob) < food_threshold_val


def train(rob):
    episode_length = 600
    learning_rate = 0.1
    discount_factor = 0.9
    epsilon = 1.0
    epsilon_decay = 0.99
    min_epsilon = 0.1
    episodes_num = 1000
    q_table_size = 1000
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
            cost = cost_function(rob)
            
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
    model = Model()
    model.load_state_dict(torch.load('/root/results/some_checkpoint_here.pth'))

    while True:
        observation = get_observation(rob)
        # print([int(o) for o in observation])
        action = model.predict(observation)
        do_action(rob, action, action_space)
        ep_len += 1

        print(ep_len)


if __name__ == "__main__":
    if sys.argv[1] == "--hardware":
        rob = HardwareRobobo(camera=True)
    elif sys.argv[1] == "--simulation":
        rob = SimulationRobobo()
        rob.play_simulation()
    else:
        raise ValueError(f"{sys.argv[1]} is not a valid argument, --simulation or --hardware expected.")
    
    # print('start test script')
    # client = RemoteAPIClient(host="localhost", port=23000)
    # print('set client')
    # sim = client.require("sim")
    # print('set simulator')
    # print(sim.getObject('/Floor'))
    
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