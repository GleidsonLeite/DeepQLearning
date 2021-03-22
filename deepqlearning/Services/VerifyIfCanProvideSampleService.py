from ..Entities.Memory import Memory


class VerifyIfCanProvideSampleService:
    @staticmethod
    def execute(memory: Memory, batch_size) -> bool:
        current_memory_length = len(memory.memories)
        can_provide_sample = current_memory_length >= batch_size
        return can_provide_sample
