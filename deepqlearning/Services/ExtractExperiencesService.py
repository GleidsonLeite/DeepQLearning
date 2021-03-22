from ..Entities.Experience import Experience
from typing import List, Tuple
import numpy as np


class ExtractExperiencesService:
    @staticmethod
    def execute(
        experiences: List[Experience],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        states = np.vstack([experience.state for experience in experiences])
        actions = np.vstack([experience.action for experience in experiences])
        rewards = np.vstack([experience.reward for experience in experiences])
        next_states = np.vstack([experience.next_state for experience in experiences])
        dones = np.vstack([experience.done for experience in experiences]).astype(
            np.uint8
        )
        return (states, actions, rewards, next_states, dones)