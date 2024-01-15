import torch
import gymnasium as gym
from cartpole_nn import Policy
# from ppo import ActorCritic
from cartpole_ppo import ActorCritic

# Load the model

policy = ActorCritic()
policy.load_state_dict(torch.load('ppo_cartpole.pth'))

env = gym.make('CartPole-v1', render_mode='human')

state = env.reset()[0]
done = False

while not done:
    env.render()
    state = torch.tensor(state, dtype=torch.float32).reshape(1, -1)
    action_probs, _ = policy(torch.tensor(state, dtype=torch.float32))
    action = torch.multinomial(action_probs, 1).item()
    next_state, _, done, _, _ = env.step(action)
    state = next_state

env.close()