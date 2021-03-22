from .ExplorationStrategy import ExplorationStrategy


class Agent:
    def __init__(self, strategy: ExplorationStrategy, number_of_actions: int):
        self.current_step = 0
        self.exploration_strategy = strategy
        self.number_of_actions = number_of_actions