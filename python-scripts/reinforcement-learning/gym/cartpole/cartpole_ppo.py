import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

# PPO: Improve the policy while ensuring the new policy doesn't deviate too far from the old one.

# Hyperparams
GAMMA = 0.99 # Importance of rewards
EPS_CLIP = 0.2 # Provides a threshold for how much the policy can change in a single training update
LR = 0.0003
BATCH_SIZE = 32
K_EPOCHS = 4 # Defines how many times algorithm will iterate over all the experiences per iteration of training

# Create NN for policy and value
class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        self.fc = nn.Linear(4, 128)
        self.policy_layer = nn.Linear(128, 2) # 2 actions to choose from (Actor)
        self.value_layer = nn.Linear(128, 1) # Estimates value of state, predicting expected return from state under current policy (Critic)

    def forward(self, x):
        x = torch.relu(self.fc(x))
        policy = torch.softmax(self.policy_layer(x), dim=1) # dim=1 is a necessary evil where we must specify the axis which softmax must be applied to
        # The value layer assigns a single value to the state. This value represents the expected return (sum of rewards) form the current state
        value = self.value_layer(x) 
        return policy, value

def compute_returns(rewards, masks, gamma):
    returns = []
    R = 0
    for reward, mask in zip(reversed(rewards), reversed(masks)):
        R = reward + gamma * R * mask
        returns.insert(0, R)
    return returns

def ppo_update(policy, optimizer, memory, ppo_epochs, batch_size, clip_param):
    for _ in range(ppo_epochs):
        for states, actions, old_log_probs, returns, advantage in memory.sample(batch_size):
            # Evaluate new policy
            new_log_probs, state_values = policy.evaluate(states, actions)

            # Calculate the advantage
            #! Need to understand PPO better before implementing this



    
def main():
    # Create instance of cartpole env
    env = gym.make('CartPole-v1', render_mode='rgb_array')

    # Create instance of local policy network and define optimizer
    policy = Policy()
 
    # Adam creates separate learning rates for each param, 
    # adapting based on the mean and variance of past gradients
    optimizer = optim.Adam(policy.parameters(), lr=0.01)

    # Training loop
    for episode in range(10000):
        state = env.reset()[0]
        terminated = False
        truncated = False
        t= 0 #timesteps
        while not terminated or truncated:
            # The state is read from the environment and is converted into a pytorch tensor
            # The action is determined using only state observations
            state = state.reshape(1, -1)
            action_probs = policy(torch.tensor(state, dtype=torch.float32))
            # multinomial distributions model the likelihood of several different things happening
            # the 1 refers to picking one action from the distribution (highest probability)
            action = torch.multinomial(action_probs, 1).item()
            # The reward function is in the environment in this case
            next_state, reward, terminated, truncated, _ = env.step(action)

            # Train
            optimizer.zero_grad() # Resets all gradients of all model params to zero
            # Calculate loss (negative reward for CartPole as it's a maximization task)
            # Taking the negative log probability of the chosen action encourages the model to increase
            # the prob of actions that receive higher rewards. Negative log prob of highly likely actions
            # will be small. The loss being scaled by the reward tells the model that the probability
            # of the chosen action should have been higher/lower
            loss = -torch.log(action_probs[0, action]) * reward
            loss.backward() # compute the gradients
            optimizer.step() # update params based on gradients

            state = next_state
            t += 1
        end_condition = ''
        if terminated:
            end_condition = 'terminated'
        else:
            end_condition = 'truncated'

        print(f'Episode {episode + 1} {end_condition} after {t+1} timesteps')

    # Save the model
    torch.save(policy.state_dict(), 'cartpole.pth')

if __name__ == '__main__':
    main()