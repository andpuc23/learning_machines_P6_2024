#!/usr/bin/env python3
from robobo_interface import SimulationRobobo, IRobobo
import numpy as np
import torch
import cv2

from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

import os
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

S_SIZE = 12 # 10 numbers per observation: 8 irs sensors + number of target pixels in 4 quadrants
A_SIZE = 4 # 4 actions - forward, right, left, back
LAST_FOOD_COLLECTED:int = 0
POSSIBLE_ACTIONS = ['move_forward', 'turn_right', 'turn_left', 'move_back']

def get_number_of_target_pixels(img):
    blue, green, red = cv2.split(img)
    px_num = blue.shape[0]*blue.shape[1]
    mask = (green > blue+10) & (green > red+10)
    count = np.count_nonzero(mask)
    return (count / px_num)

def get_reward_for_food(rob:IRobobo, action):
    global LAST_FOOD_COLLECTED
    food_collected = LAST_FOOD_COLLECTED
    if rob.nr_food_collected() > food_collected:
        LAST_FOOD_COLLECTED = LAST_FOOD_COLLECTED + 1
        # print("i'm here now, collected food:", LAST_FOOD_COLLECTED)
        return 100
    else:
        # print("got here for some reason")
        return 0


def get_reward(rob, action, t):
    image = rob.get_image_front()
    cv2.imwrite("/root/results/picture.jpeg", image) 

    top_half = get_number_of_target_pixels(image[:image.shape[0]//2, :, :])
    bottom_half = get_number_of_target_pixels(image[image.shape[0]//2:, :, :])

    #pixels = get_number_of_target_pixels(image)
    obstacles = (np.clip(max(rob.read_irs()), 0, 1000) / 1000)
    food = get_reward_for_food(rob, action)

    orient = rob.read_wheels()
    ori = abs(orient.wheel_pos_l - orient.wheel_pos_r) / (100*(t+1))
    
    if food > 0:
        reward = food
    elif bottom_half+top_half < 0.1:
        reward = -2 - ori
    elif obstacles > 0.3:
        reward = (-3 * obstacles) - ori
    else:
        reward = 2*top_half + bottom_half
    
    #reward -= ori

    #print(t,"total food:", LAST_FOOD_COLLECTED, "pixels:", round(bottom_half,3), "food:", food, "obstacles:", round(obstacles,3), "ori:",round(ori,3), "reward:", round(reward,3))
    return reward


def get_number_of_tgt_px_quad(img):
    top_left = get_number_of_target_pixels(img[:img.shape[0]//2, :img.shape[1]//2, :])
    top_right = get_number_of_target_pixels(img[:img.shape[0]//2, img.shape[1]//2:, :])
    bottom_left = get_number_of_target_pixels(img[img.shape[0]//2:, :img.shape[1]//2, :])
    bottom_right = get_number_of_target_pixels(img[img.shape[0]//2:, img.shape[1]//2:, :])
    return top_left, top_right, bottom_left, bottom_right


def get_observation(rob:IRobobo):
    observation = rob.read_irs() + list(get_number_of_tgt_px_quad(rob.get_image_front()))
    return observation, rob.get_image_front()


def do_action(rob:IRobobo, action): 
    if action not in POSSIBLE_ACTIONS:
        print('do_action(): action unknown:', action)
        return 0
    block = 0
    if action == 'move_forward':
        block = rob.move(50, 50, 500)
    elif action == 'turn_right':
        block = rob.move(20, -20, 250)
    elif action == 'turn_left':
        block = rob.move(-20, 20, 250)
    else:
        block = rob.move(-20, -20, 500)
    return block


class Policy(nn.Module):
    def __init__(self, s_size, a_size, h_size):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(s_size, h_size)
        self.fc2 = nn.Linear(h_size, a_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)
    
    def act(self, state):
        state = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(device)
        probs = self.forward(state).cpu()
        try:
            m = Categorical(probs)
        except: # probs are [NaN, NaN, NaN, NaN]
            m = Categorical([.25, .25, .25, .25])
        action = m.sample()
        return action.item(), m.log_prob(action)

def get_simulation_done(rob:IRobobo):
    global LAST_FOOD_COLLECTED
    return LAST_FOOD_COLLECTED == 7


def reset_food(rob):
    global LAST_FOOD_COLLECTED
    LAST_FOOD_COLLECTED = 0


class PolicyGradientModel:
    def __init__(self, rob, hidden_size):
        self.rob = rob
        # self.init_position = self.rob.get_position()
        # self.init_orientation = self.rob.read_orientation()
        # print('position', self.init_position)
        # print('orientation', self.init_orientation)
        self.policy = Policy(S_SIZE, A_SIZE, hidden_size)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=0.01)


    def _reinforce(self, policy, optimizer, n_training_episodes, max_t, gamma, print_every):
        scores_deque = deque(maxlen=print_every)
        scores = []

        for i_episode in tqdm(range(1, n_training_episodes+1)):
            saved_log_probs = []
            rewards = []
        
            state = get_observation(self.rob)[0]
            for t in range(max_t):
                action, log_prob = policy.act(state)
                saved_log_probs.append(log_prob)
                
                block = do_action(self.rob, POSSIBLE_ACTIONS[action])
                # state, reward, done = get_observation(self.rob)[0], get_reward(self.rob, t, action), get_simulation_done(self.rob)
                #   get_reward_for_food(self.rob, action)+get_reward(self.rob, action),\
                state, reward, done = get_observation(self.rob)[0], get_reward(self.rob, action, t), get_simulation_done(self.rob)
                
                if done:
                    # self.rob.stop_simulation()
                    # self.rob.set_position(self.init_position, self.init_orientation)
                    # self.rob.play_simulation()
                    break

                rewards.append(reward)
                self.rob.is_blocked(block)

            self.rob.stop_simulation()
            # self.rob.set_position(self.init_position, self.init_orientation)
            reset_food(self.rob)
            self.rob.play_simulation()
            self.rob.set_phone_tilt(110, 50)

            print('sum of rewards', sum(rewards))
            scores_deque.append(sum(rewards))
            scores.append(sum(rewards))
            
            returns = deque(maxlen=max_t) 
            n_steps = len(rewards) 
            
            for t in range(n_steps)[::-1]:
                disc_return_t = (returns[0] if len(returns)>0 else 0)
                returns.appendleft( gamma*disc_return_t + rewards[t]   )    
                
            eps = np.finfo(np.float32).eps.item()
            
            returns = torch.tensor(returns)
            returns = (returns - returns.mean()) / (returns.std() + eps)
            policy_loss = []
            for log_prob, disc_return in zip(saved_log_probs, returns):
                policy_loss.append(-log_prob * disc_return)
            # print(policy_loss)
            policy_l = torch.cat([p for p in policy_loss if p.shape != torch.Size([])]).sum()
            
            optimizer.zero_grad()
            policy_l.backward()
            optimizer.step()

            # print("scores_deque:", scores_deque, "rewards:", rewards, "sum:", sum(rewards))
            
            if i_episode % print_every == 0:
                name = f"/root/results/task_2_policy_{i_episode}.pth"
                print('Episode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
                self.save_model(name)
            
        return scores

    def train(self, num_episodes, max_t, gamma, print_every=10):
        scores = self._reinforce(self.policy, self.optimizer, print_every=print_every,
                                 n_training_episodes=num_episodes,
                                 max_t=max_t, gamma=gamma)
        return scores

    def predict(self, state):
        predicted_action, proba = self.policy.act(state)
        return POSSIBLE_ACTIONS[predicted_action], proba


    def __evaluate_agent(env, max_steps, n_eval_episodes, policy):
      """
      Evaluate the agent for ``n_eval_episodes`` episodes and returns average reward and std of reward.
     :param env: The evaluation environment
      :param n_eval_episodes: Number of episode to evaluate the agent
      :param policy: The Reinforce agent
      """
      episode_rewards = []
      for episode in range(n_eval_episodes):
        state = env.reset()
        step = 0
        done = False
        total_rewards_ep = 0
    
        for step in range(max_steps):
          action, _ = policy.act(state)
          new_state, reward, done, info = env.step(action)
          total_rewards_ep += reward
        
          if done:
            break
          state = new_state
        episode_rewards.append(total_rewards_ep)
      mean_reward = np.mean(episode_rewards)
      std_reward = np.std(episode_rewards)

      return mean_reward, std_reward

    def save_model(self, path):

        if os.path.exists(path):
            print(f"Policy gradient model save() WARN: file {path} exists, saving under {path.split('.')[0]}(1)")
        torch.save(self.policy.state_dict(), path)
        print(f"Policy gradient model save() INFO: saved under {path}")


def train(rob:IRobobo):
    model = PolicyGradientModel(rob, 16)
    # model.policy.load_state_dict(torch.load('/root/results/policy_5.pth'))
    print('INFO set up model, starting training')
    model.train(100, max_t=300, gamma=0.7, print_every=5)
    model.save_model('./results/100_epochs.pth')


def run(rob:IRobobo):
    model = PolicyGradientModel(rob, 16)
    # model.policy.load_state_dict(torch.load('/root/results/go_forward_starting_position.pth'))
    print('INFO loaded model from checkpoint')

    reward = 0
    max_iter = 1000
    iter = 0
    while iter < max_iter:
        observation = get_observation(rob)

        action, prob = model.predict(observation[0])
        print(f'INFO action: {action}, probability: {prob}')
        do_action(rob, action)
       # reward += get_reward(rob,action)
        iter += 1
        if iter % 50 == 0:
            print('reward:', reward)

def run_task_2(rob):
    if isinstance(rob, SimulationRobobo):
        rob.play_simulation()
    print('INFO started simulation')

    rob.set_phone_tilt(110, 50)
    
    train(rob)
    #run(rob)

    if isinstance(rob, SimulationRobobo):
        rob.stop_simulation()


if __name__ == "__main__":
    rob = SimulationRobobo()
    run_task_2(rob)