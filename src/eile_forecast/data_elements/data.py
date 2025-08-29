from abc import ABC, abstractmethod


class Data(ABC):
    cfg = None

    def __init__(self, cfg) -> None:
        self.cfg = cfg

    def get_config(self):
        if self.cfg is None:
            raise ValueError("Configuration has not been initialized.")
        return self.cfg

    # @lru_cache(maxsize=1)

    @abstractmethod
    def generate_data(self):
        pass
