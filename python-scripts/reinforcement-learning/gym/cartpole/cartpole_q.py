import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import sys
sys.path.append('../')
from qpolicy import Q

#! This file was abandoned as creating a discretized state space for Q-learning seemed suboptimal for the given task
    
def main():
    # Create instance of cartpole env
    env = gym.make('CartPole-v1', render_mode='rgb_array')

    num_states = env.observation_space.shape[0]
    
    num_actions = env.action_space.n
    policy = Q(num_states, num_actions)

    # Training loop
    for episode in range(1000):
        state = env.reset()[0]
        terminated = False
        truncated = False
        t= 0 #timesteps
        while not terminated or truncated:
           
            state = state.reshape(1, -1)[0]
            print(state)
            action = policy.choose_action(state)
           
            next_state, reward, terminated, truncated, _ = env.step(action)

            # Train

            policy.update(state, action, reward, next_state)

            state = next_state
            t += 1
        end_condition = ''
        if terminated:
            end_condition = 'terminated'
        else:
            end_condition = 'truncated'

        print(f'Episode {episode + 1} {end_condition} after {t+1} timesteps')

    # Save the model
    torch.save(policy.state_dict(), 'cartpole_q.pth')

if __name__ == '__main__':
    main()