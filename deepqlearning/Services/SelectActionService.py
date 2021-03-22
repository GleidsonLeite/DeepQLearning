from ..Entities.Agent import Agent
from ..Entities.Model import Model
from .GetExplorationRateService import GetExplorationRateService
import numpy as np
from random import random, randrange


class SelectActionService:
    @staticmethod
    def execute(agent: Agent, state: np.ndarray, policy_net: Model) -> int:
        current_step = agent.current_step
        exploration_strategy = agent.exploration_strategy
        exploration_rate = GetExplorationRateService.execute(
            exploration_strategy=exploration_strategy, current_step=current_step
        )

        should_agent_explore = exploration_rate > random()

        if should_agent_explore:
            action = randrange(agent.number_of_actions)
            return action
        else:
            formatted_state = np.atleast_2d(state).astype("float32")
            output_from_model = policy_net(formatted_state)
            action = np.argmax(output_from_model)
            return action