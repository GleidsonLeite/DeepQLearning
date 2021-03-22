from .ExtractExperiencesService import ExtractExperiencesService
from ..Entities.Experience import Experience
from ..Entities.Model import Model

from typing import List
import numpy as np
import tensorflow as tf


class DeepDoubleQLearningStepService:
    @staticmethod
    def execute(
        bath_of_experiences: List[Experience],
        target_net: Model,
        policy_net: Model,
        gamma: float,
        number_of_actions: int,
        optimizer: tf.optimizers.Optimizer,
    ) -> float:
        (
            states,
            actions,
            rewards,
            next_states,
            dones,
        ) = ExtractExperiencesService.execute(bath_of_experiences)
        formatted_next_states = np.atleast_2d(next_states).astype(np.float32)
        formatted_rewards = np.atleast_2d(rewards).astype(np.float32)
        output_from_policy_net = policy_net(formatted_next_states)
        argmax_from_policy_net_output = np.argmax(output_from_policy_net, axis=1)
        output_from_target_net = target_net(formatted_next_states)
        actions_in_one_hot = tf.one_hot(
            np.squeeze(argmax_from_policy_net_output), number_of_actions
        )

        actions_gathered = tf.math.reduce_sum(
            output_from_target_net * actions_in_one_hot, axis=1, keepdims=True
        )

        q_s_a_prime = formatted_rewards + gamma * actions_gathered * (1 - dones)
        q_s_a_prime_tensor = tf.convert_to_tensor(q_s_a_prime, dtype=tf.float32)
        with tf.GradientTape() as tape:
            formatted_states = np.atleast_2d(states).astype(np.float32)
            formatted_actions = np.squeeze(actions).astype(np.float32)
            output_from_policy_net = policy_net(formatted_states)
            actions_in_one_hot = tf.one_hot(formatted_actions, number_of_actions)
            q_s_a = tf.math.reduce_sum(
                output_from_policy_net * actions_in_one_hot, axis=1, keepdims=True
            )
            loss = tf.math.reduce_mean(tf.square(q_s_a_prime_tensor - q_s_a))

        policy_net_variables = policy_net.trainable_variables
        gradients = tape.gradient(loss, policy_net_variables)
        optimizer.apply_gradients(zip(gradients, policy_net_variables))
        return loss.numpy()
