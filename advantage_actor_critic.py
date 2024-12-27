from actor_critic import ActorCritic


class AdvantageActorCritic(ActorCritic):
    def __init__(self, mdp, actor, critic):
        super().__init__(mdp, actor, critic)

    def update_actor(self, rewards, states, actions, next_states, next_actions, dones):

        state_values = self.critic.get_values(states)
        next_state_values = self.critic.get_values(next_states)

        advantages = [
            reward + (self.mdp.get_discount_factor() * next_state_value * (1 - done)) - state_value
            for reward, state_value, next_state_value, done in zip(
                rewards, state_values, next_state_values, dones
            )
        ]

        self.actor.update(states, actions, advantages)

    def update_critic(self, reward, state, action, next_state, next_action, done):
        state_value = self.critic.get_value(state)
        next_state_value = self.critic.get_value(next_state)
        delta = (
            reward + self.mdp.get_discount_factor() * next_state_value * (1 - done) - state_value
        )
        self.critic.update(state, delta)
