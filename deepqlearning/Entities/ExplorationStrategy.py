class ExplorationStrategy:
    def __init__(self, start: float, end: float, decay_rate: float):
        self.start = start
        self.end = end
        self.decay_rate = decay_rate