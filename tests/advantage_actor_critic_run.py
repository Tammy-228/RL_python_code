from advantage_actor_critic import AdvantageActorCritic
from deep_nn_policy import DeepNeuralNetworkPolicy
from deep_value_function import DeepValueFunction
from tabular_value_function import TabularValueFunction
from gridworld import GridWorld
from plot import Plot

gridworld = GridWorld(noise=0.01)

# Instantiate the critic
critic = DeepValueFunction(state_space=len(gridworld.get_initial_state()), hidden_dim=16)
#critic = TabularValueFunction()

# Instantiate the actor
state_space = len(gridworld.get_initial_state())
action_space = len(gridworld.get_actions())
actor = DeepNeuralNetworkPolicy(state_space, action_space)

advantage_actor_critic = AdvantageActorCritic(mdp=gridworld, actor=actor, critic=critic)
#gridworld.visualise_value_function(critic, grid_size=0.8, title=f"Value Function: {0} iterations")
#gridworld.visualise_stochastic_policy(actor)
#gridworld.visualise_policy_as_image(actor)

#advantage_actor_critic.execute(100)
#gridworld.visualise_value_function(critic, grid_size=0.8, title=f"Value Function: {100} iterations")
#gridworld.visualise_stochastic_policy(actor)
#gridworld.visualise_policy_as_image(actor)

rewards = advantage_actor_critic.execute(200)
#gridworld.visualise_value_function(critic, grid_size=0.8, title=f"Value Function: {1000} iterations")
#gridworld.visualise_stochastic_policy(actor)
#gridworld.visualise_policy_as_image(actor)

print(gridworld.value_function_to_string(critic))
print(gridworld.stochastic_policy_to_string(actor))

Plot.plot_cumulative_rewards(["AAC"], [rewards], smoothing_factor=0.8)
