import sys
from robobo_interface import SimulationRobobo, HardwareRobobo
from task1_model import Model
from collections import deque
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim


action_space = ['move_forward', 'move_backward', 'turn_left', 'turn_right']
num_episodes = 10_000 
collision_threshold = 100 # sensor reading to stop episode for collision
print_every = 10 # print results and save every X episodes
training = True
max_time = 100 # max number of actions per episode
gamma = 0.9 # discount factor


def do_action(rob, action_idx, action_space):
    action = action_space[action_idx]
    if action == 'move_forward':
        rob.move_blocking(25, 25, 250)
    elif action == 'move_backward':
        rob.move_blocking(-25, -25, 250)
    elif action == 'turn_left':
        rob.move_blocking(-25, 25, 250)
    elif action == 'turn_right':
        rob.move_blocking(25, -25, 250)
    else:
        print('unknown action:', action_idx)


def collided(rob) -> bool:
    return any([r > collision_threshold for r in rob.read_irs()])


def target_function(rob, action):
    # we encourage robot to move forward, but also ok with other actions
    # a thing here is that he may go forward-backward all the time
    if collided(rob):
        return -100
    if action == 0:
        return 1
    if action == 1:
        return 0.3
    if action == 2 or action == 3:
        return 0.5 


if __name__ == "__main__":
    if sys.argv[1] == "--hardware":
        rob = HardwareRobobo(camera=True)
    elif sys.argv[1] == "--simulation":
        rob = SimulationRobobo()
    else:
        raise ValueError(f"{sys.argv[1]} is not a valid argument, --simulation or --hardware expected.")
    
    if sys.argv[2] == '--train':
        training = True
    elif sys.argv[2] == '--test':
        training = False
    else:
        raise ValueError(f"{sys.argv[2]} is not a valid argument, --train or --test expected.")

    print(f'running {"training" if training else "inference"} in {"real world" if isinstance(rob, HardwareRobobo)else "simulation"}')

    model = Model(training, action_space)
    if training:

        optimizer = optim.Adam(model.parameters(), lr=0.01)
        scores_deque = deque(maxlen=print_every)
        scores = []
        init_position = rob.position()
        init_orientation = rob.read_orientation()

        for episode in range(1, num_episodes+1):
            rewards = []
            saved_logprobs = []
            if isinstance(rob, SimulationRobobo):
                rob.play_simulation()
            observation = rob.read_irs()
            for t in range(max_time):
                action_i, proba = model.predict(observation)
                saved_logprobs.append(proba)

                do_action(rob, action_i, action_space)

                observation, reward, done = \
                    rob.read_irs(), target_function(rob, action_i), collided(rob)
                
                if done:
                    rob.stop_simulation()
                    rob.set_position(init_position, init_orientation)
                    rob.play_simulation()
                    break
                rewards.append(reward)

            rob.stop_simulation()
            scores_deque.append(sum(rewards))
            scores.append(sum(rewards))
            
            returns = deque(maxlen=max_time)
            n_steps = len(rewards)

            for t in range(n_steps)[::-1]:
                dicounted_return_t = returns[0] if returns else 0
                returns.appendleft(gamma*dicounted_return_t + rewards[t])
            
            eps = np.finfo(np.float32).eps.item()

            returns = torch.tensor(returns)
            returns = (returns - returns.mean()) / (returns.std() + eps)
            policy_loss = []

            for log_prob, dicounted_return in zip(saved_logprobs, returns):
                policy_loss.append(-log_prob*dicounted_return)
            policy_loss = torch.cat(policy_loss).sum()

            optimizer.zero_grad()
            policy_loss.backward()
            optimizer.step()

            if episode % print_every == 0: 
                # todo plot graphs
                print(f'Episode {episode}\tAverage Score: {np.mean(scores_deque):.2f}')
                model.save_checkpoint(f'/root/results/task1_checkpoint_{episode}.pth')

    else:
        while not collided(rob):
            observation = rob.read_irs()
            action, _ = model.predict(observation)
            do_action(rob, action, action_space)



    if isinstance(rob, SimulationRobobo):
        rob.stop_simulation()