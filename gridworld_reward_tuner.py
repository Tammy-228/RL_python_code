from gridworld_potential_function import GridWorldPotentialFunction


class GridWorldRewardTuner(GridWorldPotentialFunction):
    def get_tuning(self, state, next_state, discount_factor):
        return discount_factor * super().get_potential(
            next_state
        ) - super().get_potential(state)
