import torch

class Memory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.is_terminals = []
        self.returns = []

    def clear(self):
        del self.states[:]
        del self.actions[:]
        del self.log_probs[:]
        del self.rewards[:]
        del self.is_terminals[:]
        del self.returns[:]

    def push(self, state, action, log_prob, reward, is_terminal):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.is_terminals.append(is_terminal)

    def __len__(self):
        return len(self.states)

    def sample(self, batch_size):
        # Sample a batch of experiences from the memory
        batch_start = 0
        while batch_start < len(self.states):
            batch_end = min(batch_start + batch_size, len(self.states))
            states = torch.cat(self.states[batch_start:batch_end], dim=0)
            actions = torch.cat(self.actions[batch_start:batch_end], dim=0)
            old_log_probs = torch.cat(self.log_probs[batch_start:batch_end], dim=0)
            returns = torch.cat(self.returns[batch_start:batch_end], dim=0)
            advantages = returns - returns.mean()
            yield states, actions, old_log_probs, returns, advantages
            batch_start = batch_end
