from gridworld_potential_function import GridWorldPotentialFunction

class GridWorldNonPotentialFunction(GridWorldPotentialFunction):
    def get_tuning(self, state, next_state, discount_factor):
        default = discount_factor * super().get_potential(next_state) - super().get_potential(state)
        tune_states = [(0,0), (1,0), (2,0), (2,1), (2,2), (1,2), (0,2), (0,1)]
        for s,s_next in zip(tune_states, tune_states[1:] + [tune_states[0]]):
            if state == s and next_state == s_next:
                default += 0.05
        return default
