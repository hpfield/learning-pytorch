import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear_memory(self):
        # Empties the given list without deleting the list itself
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(4, 128)
        self.actor = nn.Linear(128, 2)
        self.critic = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        action_prob = torch.softmax(self.actor(x), dim=1)
        state_values = self.critic(x)
        return action_prob, state_values
    
def ppo_update(policy, optimizer, memory, gamma, eps_clip, K_epochs):
    # Helper function to compute discounted rewards
    def discounted_rewards(rewards, gamma, is_terminals):
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(rewards), reversed(is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (gamma * discounted_reward)
            yield discounted_reward

    # Compute discounted rewards and normalize them
    rewards = list(discounted_rewards(memory.rewards, gamma, memory.is_terminals))
    rewards = torch.tensor(rewards[::-1], dtype=torch.float32).detach()
    rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

    # Convert list to tensor
    old_states = torch.squeeze(torch.stack(memory.states, dim=0)).detach()
    old_actions = torch.tensor(memory.actions).detach()
    old_logprobs = torch.tensor(memory.logprobs).detach()

    # Optimize policy for K epochs
    for _ in range(K_epochs):
        # Evaluating old actions and values
        logprobs, state_values = policy(old_states)
        state_values = torch.squeeze(state_values)

        # Match state_values tensor dimensions with rewards tensor
        if len(state_values) != len(rewards):
            state_values = state_values[:len(rewards)]

        # Finding the ratio (pi_theta / pi_theta__old)
        new_logprobs = logprobs.gather(1, old_actions.unsqueeze(1)).squeeze(1)
        ratios = torch.exp(new_logprobs - old_logprobs.detach())

        # Finding Surrogate Loss
        advantages = rewards - state_values.detach()
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1 - eps_clip, 1 + eps_clip) * advantages
        loss = -torch.min(surr1, surr2) + 0.5 * (rewards - state_values)**2

        # Take gradient step
        optimizer.zero_grad()
        loss.mean().backward()
        optimizer.step()

    # Clear memory
    memory.clear_memory()


def main():
    env = gym.make('CartPole-v1', render_mode='rgb_array')
    policy = ActorCritic()
    optimizer = optim.Adam(policy.parameters(), lr=0.002)
    memory = Memory()

    # Hyperparameters
    gamma = 0.99
    eps_clip = 0.2
    update_timestep = 2000
    max_episodes = 10000
    K_epochs = 4

    timestep = 0
    for episode in range(max_episodes):
        state = env.reset()[0]
        for t in range(1, 10000):  # Set a large number for timesteps
            timestep += 1
            state = torch.tensor(state, dtype=torch.float32).reshape(1, -1)
            action_probs, _ = policy(torch.tensor(state, dtype=torch.float32))
            action = torch.multinomial(action_probs, 1).item()

            memory.states.append(state)
            memory.actions.append(action)
            memory.logprobs.append(torch.log(action_probs[0, action]))

            state, reward, terminated, truncated, _ = env.step(action)
            memory.rewards.append(reward)
            memory.is_terminals.append(terminated)

            if timestep % update_timestep == 0:
                # PPO update
                # Implement the PPO update function here
                ppo_update(policy, optimizer, memory, gamma, eps_clip, K_epochs)
                pass

            if terminated:
                break

        # Logging
        if episode % 100 == 0:
            print(f'Episode {episode + 1} finished')

    torch.save(policy.state_dict(), 'ppo_generated.pth')

if __name__ == '__main__':
    main()
