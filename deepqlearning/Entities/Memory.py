from collections import deque
from typing import Deque
from .Experience import Experience


class Memory:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.memories: Deque[Experience] = deque(maxlen=capacity)