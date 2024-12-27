from itertools import count

from model_free_learner import ModelFreeLearner

class ActorCritic(ModelFreeLearner):
    def __init__(self, mdp, actor, critic):
        self.mdp = mdp
        self.actor = actor  # Actor (policy based) to select actions
        self.critic = critic  # Critic (value based) to evaluate actions

    def execute(self, episodes=100, max_episode_length=float('inf')):
        episode_rewards = []
        for episode in range(episodes):
            actions = []
            states = []
            rewards = []
            next_states = []
            next_actions = []
            dones = []

            state = self.mdp.get_initial_state()
            action = self.actor.select_action(state, self.mdp.get_actions(state))
            episode_reward = 0.0
            for step in count():
                (next_state, reward, done) = self.mdp.execute(state, action)
                next_action = self.actor.select_action(next_state, self.mdp.get_actions(next_state))
                self.update_critic(reward, state, action, next_state, next_action, done)

                # Store the information from this step of the trajectory
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                next_states.append(next_state)
                next_actions.append(next_action)
                dones.append(done)

                state = next_state
                action = next_action
                episode_reward += reward * (self.mdp.discount_factor ** step)

                if done or step == max_episode_length:
                    break

            self.update_actor(rewards, states, actions, next_states, next_actions, dones)

            episode_rewards.append(episode_reward)

        return episode_rewards

    """ Update the actor using a batch of transitions """

    def update_actor(self, rewards, states, actions, next_states, next_actions, dones):
        abstract

    """ Update the critic """

    def update_critic(self, reward, state, action, next_state, next_action, done):
        abstract
