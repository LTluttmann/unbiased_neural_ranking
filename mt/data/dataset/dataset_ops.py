from abc import ABC, abstractmethod


class BaseDatasetOp(ABC):

    def __init__(self, on_batch:bool = True) -> None:
        super().__init__()
        self.on_batch = on_batch

    @abstractmethod
    def call_on_batch(self):
        pass

    @abstractmethod
    def call_on_single_example(self):
        pass


    def __call__(self, *args, **kwargs):
        if self.on_batch:
            return self.call_on_batch(*args, **kwargs)
        else:
            return self.call_on_single_example(*args, **kwargs)
    