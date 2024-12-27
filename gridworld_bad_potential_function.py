from gridworld_potential_function import GridWorldPotentialFunction
from gridworld import GridWorld


class GridWorldBadPotentialFunction(GridWorldPotentialFunction):
    def get_potential(self, state):
        return -super().get_potential(state)
