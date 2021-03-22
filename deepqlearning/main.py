from collections import deque
from .Services.DeepDoubleQLearningStepService import (
    DeepDoubleQLearningStepService,
)
from .Services.IncreaseAgentCurrentStepService import (
    IncreaseAgentCurrentStepService,
)

from .Services.DeepQLearningStepService import DeepQLearningStepService
from .Entities.Experience import Experience
from .Entities.Memory import Memory
from .Entities.ExplorationStrategy import ExplorationStrategy
from .Entities.Model import Model
from .Entities.Agent import Agent

from .Services.CopyWeightsService import CopyWeightsService
from .Services.SelectActionService import SelectActionService
from .Services.PushExperienceToMemoryService import PushExperienceToMemoryService
from .Services.VerifyIfCanProvideSampleService import VerifyIfCanProvideSampleService
from .Services.GetRandomSampleFromMemoryService import GetRandomSampleFromMemoryService
from .Services.GetExplorationRateService import GetExplorationRateService

import gym
import tensorflow as tf
import numpy as np


def main():
    environmentName = "CartPole-v1"
    env = gym.make(environmentName)

    eps_start = 1
    eps_end = 0
    eps_decay = 1e-3
    exploration_strategy = ExplorationStrategy(
        start=eps_start, end=eps_end, decay_rate=eps_decay
    )

    memory_capacity = 5000
    batch_size = 64
    memory = Memory(capacity=memory_capacity)

    number_of_states = len(env.observation_space.sample())
    number_of_actions = env.action_space.n
    hidden_units = [250, 125, 60]

    policy_net = Model(
        number_of_states=number_of_states,
        hidden_units=hidden_units,
        number_of_actions=number_of_actions,
    )
    target_net = Model(
        number_of_states=number_of_states,
        hidden_units=hidden_units,
        number_of_actions=number_of_actions,
    )

    learning_rate = 1e-4
    optimizer = tf.optimizers.Adam(learning_rate=learning_rate)

    CopyWeightsService.execute(
        model_to_copy=policy_net,
        model_to_past=target_net,
    )

    agent = Agent(strategy=exploration_strategy, number_of_actions=number_of_actions)
    gamma = 0.98
    epochs = 10000
    target_update = 4
    max_steps = 1000

    total_rewards = deque(maxlen=1000)

    for epoch in range(epochs):
        current_state = env.reset()
        ep_rewards = 0
        losses = []

        for timestep in range(max_steps):
            env.render()
            action = SelectActionService.execute(
                agent=agent, state=current_state, policy_net=policy_net
            )
            IncreaseAgentCurrentStepService.execute(agent=agent)
            next_state, reward, done, _ = env.step(action)
            experience = Experience(
                state=current_state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done,
            )
            PushExperienceToMemoryService.execute(memory=memory, experience=experience)
            ep_rewards += reward
            current_state = next_state

            can_provide_sample = VerifyIfCanProvideSampleService.execute(
                memory=memory, batch_size=batch_size
            )

            if can_provide_sample:
                sample_of_experiences = GetRandomSampleFromMemoryService.execute(
                    memory=memory, batch_size=batch_size
                )

                loss = DeepDoubleQLearningStepService.execute(
                    bath_of_experiences=sample_of_experiences,
                    target_net=target_net,
                    policy_net=policy_net,
                    gamma=gamma,
                    number_of_actions=number_of_actions,
                    optimizer=optimizer,
                )
                losses.append(loss)
                CopyWeightsService.execute(
                    model_to_copy=policy_net, model_to_past=target_net, tau=1e-3
                )
            else:
                losses.append(0)

            if timestep % target_update == 0:
                CopyWeightsService.execute(
                    model_to_copy=policy_net,
                    model_to_past=target_net,
                )

            if done:
                break
        total_rewards.append(ep_rewards)
        average_rewards = np.mean(total_rewards)
        current_exploration_rate = GetExplorationRateService.execute(
            exploration_strategy=exploration_strategy, current_step=agent.current_step
        )
        average_loss = np.mean(losses)
        print(
            "Epoch: {}\tStep: {}\tAverage rewards by episode: {:.3}\tEpisode Reward: {}\tAverage Loss: {}\tAgent Step: {}\tExploration Rate: {}".format(
                epoch,
                timestep,
                average_rewards,
                ep_rewards,
                average_loss,
                agent.current_step,
                current_exploration_rate,
            )
        )
    env.close()


if __name__ == "__main__":
    main()
