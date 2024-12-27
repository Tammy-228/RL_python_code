import numpy as np
from itertools import count

class PPO:
    def __init__(self, mdp, actor, critic, gamma=0.99, lam=0.95, clip_epsilon=0.2, epochs=10):
        self.mdp = mdp
        self.actor = actor
        self.critic = critic
        self.gamma = gamma
        self.lam = lam
        self.clip_epsilon = clip_epsilon
        self.epochs = epochs

    def execute(self, episodes=100, max_episode_length=float('inf')):
        episode_rewards = []
        for episode in range(episodes):
            states, actions, rewards, log_probs, values = [], [], [], [], []
            state = self.mdp.get_initial_state()
            episode_reward = 0.0

            for step in count():
                action = self.actor.select_action(state, self.mdp.get_actions(state))
                next_state, reward, done = self.mdp.execute(state, action)
                log_prob = self.actor.get_probability(state, action)
                value = self.critic.get_value(state)

                states.append(state)
                actions.append(action)
                rewards.append(reward)
                log_probs.append(log_prob)
                values.append(value)
                state = next_state
                episode_reward += reward * (self.mdp.discount_factor ** step)

                if done or step == max_episode_length:
                    break
            
            episode_rewards.append(episode_reward)
            advantages, returns = self.calculate_advantages_and_returns(rewards, values)
            self.update_actor(states, actions, log_probs, advantages)
            self.critic.update_batch(states, advantages)
        
        return episode_rewards

    def calculate_advantages_and_returns(self, rewards, values):
        values = values + [0]
        advantages = []
        gae = 0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t + 1] - values[t]
            gae = delta + self.gamma * self.lam * gae
            advantages.insert(0, gae)
        returns = [adv + val for adv, val in zip(advantages, values[:-1])]
        return advantages, returns

    def update_actor(self, states, actions, log_probs, advantages):
        states = np.array(states)
        actions = np.array(actions)
        log_probs = np.array(log_probs)
        advantages = np.array(advantages)
        
        for _ in range(self.epochs):
            new_log_probs = np.array([self.actor.get_probability(state, action) for state, action in zip(states, actions)])
            ratio = np.exp(new_log_probs - log_probs)
            surr1 = ratio * advantages
            surr2 = np.clip(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages
            deltas = -np.minimum(surr1, surr2)

            self.actor.update(states, actions, deltas)

