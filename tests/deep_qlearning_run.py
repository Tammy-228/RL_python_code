from gridworld import GridWorld
from qlearning import QLearning
from old_deep_q_function import DeepQFunction
from q_policy import QPolicy
from multi_armed_bandit.epsilon_greedy import EpsilonGreedy

gridworld = GridWorld()
action_space = len(gridworld.get_actions())
state_space = len(gridworld.get_initial_state())
qfunction = DeepQFunction(state_space, action_space)
rewards = QLearning(gridworld, EpsilonGreedy(), qfunction).execute(episodes=300)
policy = QPolicy(qfunction)
gridworld.visualise_q_function(qfunction)
gridworld.visualise_policy_as_image(policy)
