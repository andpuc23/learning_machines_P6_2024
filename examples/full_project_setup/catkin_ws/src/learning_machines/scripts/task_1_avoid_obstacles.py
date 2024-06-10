import sys
from robobo_interface import SimulationRobobo, HardwareRobobo
from task1_model import Model
from collections import Counter
import matplotlib.pyplot as plt


action_space = ['move_forward', 'move_backward', 'turn_left', 'turn_right']
num_episodes = 10_000
collision_threshold = 150
training = False


def do_action(rob, action_idx, action_space):
    action = action_space[action_idx]
    if action == 'move_forward':
        rob.move_blocking(25, 25, 150)
    elif action == 'move_backward':
        rob.move_blocking(-25, -25, 150)
    elif action == 'turn_left':
        rob.move_blocking(-25, 25, 150)
    elif action == 'turn_right':
        rob.move_blocking(25, -25, 150)
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


def get_accumulated_reward(buffer):
    counts = Counter(buffer)
    return counts[0] + 0.3*counts[1] + 0.5*(counts[2]+counts[3])


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
        accumulated_rewards = []
        for episode in range(1, num_episodes):
            buffer = []
            if isinstance(rob, SimulationRobobo):
                rob.play_simulation()
            while not collided(rob):
                observation = rob.read_irs()
                action = model.predict(observation)
                do_action(rob, action, action_space)
                buffer.append(action)
                reward = target_function(rob, action)
                # model.backward somewhere here
                if reward == -100:
                    accumulated_rewards.append(get_accumulated_reward(buffer))
                    if isinstance(rob, SimulationRobobo):
                        rob.stop_simulation()
                    break
            if episode % 100 == 0:
                print(accumulated_rewards[-100].sum()/100)
                plt.scatter(range(len(accumulated_rewards)), accumulated_rewards)
                plt.title('Rewards per episode')
                plt.savefig('/root/result/figures/task1.png')
                model.save_checkpoint(f'/root/results/task1_checkpoint_{episode}.pth')

    else:
        while not collided(rob):
            observation = rob.read_irs()
            action = model.predict(observation)
            do_action(rob, action, action_space)



    if isinstance(rob, SimulationRobobo):
        rob.stop_simulation()