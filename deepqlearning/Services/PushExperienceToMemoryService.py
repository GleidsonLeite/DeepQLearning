from ..Entities.Experience import Experience
from ..Entities.Memory import Memory


class PushExperienceToMemoryService:
    @staticmethod
    def execute(memory: Memory, experience: Experience) -> None:
        memory.memories.append(experience)