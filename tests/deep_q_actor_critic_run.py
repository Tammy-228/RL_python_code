from deep_nn_policy import DeepNeuralNetworkPolicy
from q_actor_critic import QActorCritic
from old_deep_q_function import DeepQFunction
from gridworld import GridWorld
from multi_armed_bandit.epsilon_greedy import EpsilonGreedy
from qlearning import QLearning

mdp = GridWorld()
action_space = len(mdp.get_actions())
state_space = len(mdp.get_initial_state())

# Instantiate the critic
critic = DeepQFunction(state_space, action_space)

# Instantiate the actor
actor = DeepNeuralNetworkPolicy(state_space, action_space)

#  Instantiate the actor critic agent
learner = QActorCritic(mdp, actor, critic)
episode_rewards = learner.execute(episodes=500)
print(mdp.stochastic_policy_to_string(actor))
print(mdp.q_function_to_string(critic))
#mdp.visualise_stochastic_policy(actor)
#mdp.visualise_q_function(critic)
