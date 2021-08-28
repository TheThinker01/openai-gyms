import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# The Brain of the Policy Gradient Algorithm
# The Policy Network uses a Neural Network as a function approximator for our 
class PolicyNetwork(nn.Module):
    def __init__(self, alpha, input_dims, n_actions):

        super(PolicyNetwork, self).__init__()
        
        self.fc1 = nn.Linear(*input_dims, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, n_actions)
        self.optimizer = optim.Adam(self.parameters(), lr = alpha)

        # for nvidia gpu devices
        # self.device = T.device('cuda:0' if T.cuda.is_available else 'cpu')
        self.device = 'cpu'
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

# The PolicyAgent
class PolicyGradientAgent():
    def __init__(self, lr, input_dims, gamma=0.99, n_actions = 4):
        self.lr = lr
        self.gamma = gamma
        # for storing the rewards received at each step of the episode 
        self.reward_memory = []

        # for storing the log probabilities of actions taken at each step of the episode 
        self.action_memory = []

        self.policy = PolicyNetwork(self.lr, input_dims, n_actions)

    def choose_action(self, observation):
        state = T.Tensor([observation]).to(self.policy.device)
        probs = F.softmax(self.policy.forward(state))
        # The Monte Carlo way of proceeding : chooses a action according to a Categorical Distribution
        action_probs = T.distributions.Categorical(probs)
        action = action_probs.sample()
        log_probs = action_probs.log_prob(action)
        self.action_memory.append(log_probs)

        return action.item()

    def store_rewards(self, reward):
        self.reward_memory.append(reward)

    def learn(self):
        # for calculting the total reward received at each step of the process 
        self.policy.optimizer.zero_grad()
        # G_t = R_t+1 + gamma* R_t+2 + gamma^2 * R_t+3+ ...
        G = np.zeros_like(self.reward_memory, dtype = np.float64)

        for t in  range(len(self.reward_memory)):
            G_sum = 0
            discount = 1
            for k in range(t, len(self.reward_memory)):
                G_sum+=self.reward_memory[k]*discount
                discount*=self.gamma
            G[t] = G_sum
        
        G = T.tensor(G, dtype=T.float).to(self.policy.device)

        loss = 0
        for g, logprob in zip(G, self.action_memory):
            loss += -g*logprob
        self.policy.optimizer.zero_grad()
        loss.backward()
        self.policy.optimizer.step()

        self.action_memory = []
        self.reward_memory = []


















