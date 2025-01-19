from abc import ABC, abstractmethod


class BaseLLM(ABC):
    @abstractmethod
    def build_pipe(self):
        raise NotImplementedError
    
    @abstractmethod
    def get_pipe(self):
        raise NotImplementedError
