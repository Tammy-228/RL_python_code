from actor_critic import ActorCritic


class QActorCritic(ActorCritic):
    def __init__(self, mdp, actor, critic):
        super().__init__(mdp, actor, critic)

    def update_actor(self, rewards, states, actions, next_states, next_actions, dones):
        q_values = self.critic.get_q_values(states, actions)
        next_state_q_values = self.critic.get_q_values(next_states, next_actions)

        deltas = [
            reward + (self.mdp.get_discount_factor() * next_state_q_value * (1 - done)) - q_value
            for reward, q_value, next_state_q_value, done in zip(
                rewards, q_values, next_state_q_values, dones
            )
        ]
 
    def update_critic(self, reward, state, action, next_state, next_action, done):
        state_value = self.critic.get_q_value(state, action)
        next_state_value = self.critic.get_q_value(next_state, next_action)
        delta = (
            reward + (self.mdp.get_discount_factor() * next_state_value * (1 - done)) - state_value
        )
        self.critic.update(state, action, delta)

    def batch_update_critic(self, rewards, states, actions, next_states, next_actions, dones):
        state_values = self.critic.get_q_values(states, actions)
        next_state_values = self.critic.get_q_values(next_states, next_actions)
        deltas = [
            reward + (self.mdp.get_discount_factor() * next_state_value * (1 - done)) - state_value
            for reward, state_value, next_state_value, done in zip(
                rewards, state_values, next_state_values, dones
            )
        ]
        self.critic.batch_update(states, actions, deltas)