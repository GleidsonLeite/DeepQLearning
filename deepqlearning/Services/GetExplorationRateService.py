from ..Entities.ExplorationStrategy import ExplorationStrategy
from math import exp


class GetExplorationRateService:
    @staticmethod
    def execute(
        exploration_strategy: ExplorationStrategy, current_step: float
    ) -> float:
        exploration_rate_start = exploration_strategy.start
        exploration_rate_end = exploration_strategy.end
        exploration_decay_rate = exploration_strategy.decay_rate

        exploration_rate = exploration_rate_end + (
            exploration_rate_start - exploration_rate_end
        ) * exp(-current_step * exploration_decay_rate)

        return exploration_rate