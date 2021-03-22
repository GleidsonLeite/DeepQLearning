from ..Entities.Agent import Agent


class IncreaseAgentCurrentStepService:
    @staticmethod
    def execute(agent: Agent, steps_to_increase: int = 1):
        agent.current_step += steps_to_increase