#!/usr/bin/env python3
import sys
from robobo_interface import SimulationRobobo, HardwareRobobo
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
    pixels = get_img_sectors_colors()
    return irs + list(pixels)



def _distance(rob, obj1, obj2) -> float:
    pass


def cost_function(rob):
    distance_rob_to_food = _distance(rob, rob, food)
    distance_food_to_base = _distance(rob, food, target)
    if distance_rob_to_food > food_threshold_val:
        return distance_rob_to_food + 100 # should be much (x2, x3) higher than distance_food_to_base
    else:
        return distance_rob_to_food + distance_food_to_base


def train(rob):
    pass


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