import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam
import torch.nn.functional as F

from policy import StochasticPolicy


class DeepNeuralNetworkPolicy(StochasticPolicy):
    """
    An implementation of a policy that uses a PyTorch (https://pytorch.org/) 
    deep neural network to represent the underlying policy.
    """

    def __init__(self, state_space, action_space, hidden_dim=64, alpha=0.001, stochastic=True):
        self.state_space = state_space
        self.action_space = action_space

        # Define the policy structure as a sequential neural network.
        self.policy_network = nn.Sequential(
            nn.Linear(in_features=self.state_space, out_features=hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=hidden_dim, out_features=self.action_space),
        )

        # Initialize weights using Xavier initialization and biases to zero
        self._initialize_weights()

        # The optimiser for the policy network, used to update policy weights
        self.optimiser = Adam(self.policy_network.parameters(), lr=alpha)

        # Whether to select an action stochastically or deterministically
        self.stochastic = stochastic

    def _initialize_weights(self):
        for layer in self.policy_network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

        # Ensure the last layer outputs logits close to zero
        last_layer = self.policy_network[-1]
        if isinstance(last_layer, nn.Linear):
            with torch.no_grad():
                last_layer.weight.fill_(0)
                last_layer.bias.fill_(0)

    """ Select an action using a forward pass through the network """

    def select_action(self, state, actions):
        # Convert the state into a tensor so it can be passed into the network
        state = torch.as_tensor(state, dtype=torch.float32)
        action_logits = self.policy_network(state)

        # Mark out the actions that are unavailable
        mask = torch.full_like(action_logits, float('-inf'))
        mask[actions] = 0
        masked_logits = action_logits + mask

        action_distribution = Categorical(logits=masked_logits)
        if self.stochastic:
            # Sample an action according to the probability distribution
            action = action_distribution.sample()
        else:
            # Choose the action with the highest probability
            action_probabilities = torch.softmax(masked_logits, dim=-1)
            action = torch.argmax(action_probabilities)
        return action.item()

    """ Get the probability of an action being selected in a state """

    def get_probability(self, state, action):
        state = torch.as_tensor(state, dtype=torch.float32)
        with torch.no_grad():
            action_logits = self.policy_network(state)

        # A softmax layer turns action logits into relative probabilities
        probabilities = F.softmax(input=action_logits, dim=-1).tolist()
        # Convert from a tensor encoding back to the action space
        return probabilities[action]

    def evaluate_actions(self, states, actions):
        action_logits = self.policy_network(states)
        action_distribution = Categorical(logits=action_logits)
        log_prob = action_distribution.log_prob(actions.squeeze(-1))
        return log_prob.view(1, -1)

    def update(self, states, actions, deltas):
        # Convert to tensors to use in the network
        deltas = torch.as_tensor(deltas, dtype=torch.float32)
        states = torch.as_tensor(states, dtype=torch.float32)
        actions = torch.as_tensor(actions)

        action_log_probs = self.evaluate_actions(states, actions)

        # Construct a loss function, using negative because we want to descend,
        # not ascend the gradient
        loss = -(action_log_probs * deltas).sum()
        self.optimiser.zero_grad()
        loss.backward()

        # Take a gradient descent step
        self.optimiser.step()

    def save(self, filename):
        torch.save(self.policy_network.state_dict(), filename)

    @classmethod
    def load(cls, state_space, action_space, filename):
        policy = cls(state_space, action_space)
        policy.policy_network.load_state_dict(torch.load(filename))
        return policy
