import torch as T
import torch.nn.functional as F
from networks import CriticNetwork, ActorNetwork
from noise import OUActionNoise
from replay_buffer import ReplayBuffer
import numpy as np

class Agent():
    def __init__(self, alpha, beta, input_dims, tau, n_actions, gamma=0.99,
                 max_size=1000000, fc1_dims=400, fc2_dims=300, 
                 batch_size=64):
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.alpha = alpha 
        self.beta = beta

        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        
        self.noise = OUActionNoise(mu=np.zeros(n_actions))

        self.actor = ActorNetwork(alpha, input_dims, fc1_dims, fc2_dims,
                                n_actions, 'actor')
        self.critic = CriticNetwork(beta, input_dims, fc1_dims, fc2_dims,
                                n_actions, 'critic')
        self.target_actor = ActorNetwork(alpha, input_dims, fc1_dims, fc2_dims,
                                n_actions, 'target_actor')
        self.target_critic = CriticNetwork(beta, input_dims, fc1_dims, fc2_dims,
                                n_actions, 'target_critic')

        self.update_network_parameters(tau = 1)

    def choose_action(self, observation):
        self.actor.eval()
        state = T.tensor([observation], dtype = T.float).to(self.actor.device)
        mu = self.actor.forward(state).to(self.actor.device)
        mu_prime = mu + T.tensor(self.noise(), 
                                dtype=T.float).to(self.actor.device)
        self.actor.train()

        return mu_prime.cpu().detach().numpy()[0]

    def remember(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)
    
    def save_models(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_critic.save_checkpoint()
    
    def load_models(self):
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_critic.load_checkpoint()
    
    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return
        
        states, actions, rewards, states_, dones = \
            self.memory.sample_buffer(self.batch_size)
        
        # The above function returns numpy arrays, turn into pytorch tensor
        states  = T.tensor(states, dtype=T.float).to(self.actor.device)
        actions = T.tensor(actions, dtype=T.float).to(self.actor.device)
        rewards = T.tensor(rewards, dtype=T.float).to(self.actor.device)
        states_ = T.tensor(states_, dtype=T.float).to(self.actor.device)
        # preserve the underlying datatype for the dones tensor
        dones = T.tensor(dones).to(self.actor.device)

        # first get the actions from the target actor network for the next state
        target_actions = self.target_actor.forward(states_)
        critic_value_ = self.target_critic.forward(states_, target_actions)
        critic_value  = self.critic.forward(states, actions)

        # set the critic value of all the next states which are actually terminal to 0
        critic_value_[dones] = 0.0

        # reduce one needless dimesion
        critic_value_ = critic_value_.view(-1)

        # calculate the target for the entire batch
        targets = rewards + self.gamma*critic_value_
        targets = targets.view(self.batch_size, 1)

        self.critic.optimizer.zero_grad()
        critic_loss = F.mse_loss(targets, critic_value)
        critic_loss.backward()
        self.critic.optimizer.step()

        self.actor.optimizer.zero_grad()
        actor_loss = -self.critic.forward(states, self.actor.forward(states))
        actor_loss = T.mean(actor_loss)
        actor_loss.backward()
        self.actor.optimizer.step()

        self.update_network_parameters()

    def update_network_parameters(self, tau=None):
        if tau is None :
            tau = self.tau
        
        # return a tuple of the form [(name, value)]
        actor_params  = self.actor.named_parameters()
        critic_params = self.critic.named_parameters()
        target_critic_params = self.target_critic.named_parameters()
        target_actor_params = self.target_actor.named_parameters()

        # convert to a dictionary of model params
        actor_state_dict = dict(actor_params)
        critic_state_dict = dict(critic_params)
        target_actor_state_dict = dict(target_actor_params)
        target_critic_state_dict = dict(target_critic_params)

        # Compute the new values for the parameters using soft update
        for name in critic_state_dict:
            critic_state_dict[name] = tau*critic_state_dict[name].clone() + \
                                    (1-tau)*target_critic_state_dict[name].clone()

        for name in actor_state_dict:
            actor_state_dict[name] = tau*actor_state_dict[name].clone() + \
                                    (1-tau)*target_actor_state_dict[name].clone()

        # finally load these new state_dicts into the target networks, 
        #    thus updating them.
        self.target_critic.load_state_dict(critic_state_dict)
        self.target_actor.load_state_dict(actor_state_dict)

