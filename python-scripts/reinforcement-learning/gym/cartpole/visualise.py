import torch
import gymnasium as gym
from cartpole import Policy

# Load the model

policy = Policy()
policy.load_state_dict(torch.load('cartpole.pth'))

env = gym.make('CartPole-v1', render_mode='human')

state = env.reset()[0]
done = False

while not done:
    env.render()
    state = state.reshape(1, -1)
    action_probs = policy(torch.tensor(state, dtype=torch.float32))
    action = torch.argmax(action_probs).item()
    next_state, _, done, _, _ = env.step(action)
    state = next_state

env.close()