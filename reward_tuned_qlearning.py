from qlearning import QLearning


class RewardTunedQLearning(QLearning):
    def __init__(self, mdp, bandit, tuner, qfunction):
        super().__init__(mdp, bandit, qfunction=qfunction)
        self.tuner = tuner

    def get_delta(self, reward, state, action, next_state, next_action):
        q_value = self.qfunction.get_q_value(state, action)
        next_state_value = self.state_value(next_state, next_action)
        tuning = self.tuner.get_tuning(state, next_state, self.mdp.discount_factor)
        delta = reward + tuning + self.mdp.discount_factor * next_state_value - q_value
        return delta