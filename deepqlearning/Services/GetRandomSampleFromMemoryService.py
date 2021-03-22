from ..Entities.Experience import Experience
from typing import List
from ..Entities.Memory import Memory
import random


class GetRandomSampleFromMemoryService:
    @staticmethod
    def execute(memory: Memory, batch_size: int) -> List[Experience]:
        memories = memory.memories
        randomSample = random.sample(memories, batch_size)
        return randomSample