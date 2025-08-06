from typing import Callable
from memory import MemoryWorker  # import your MemoryWorker
from functools import wraps


def auto_save_result(memoryworker: MemoryWorker):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            result, desc = func(*args, **kwargs)
            memoryworker.add_object(result, desc)
            return result, memoryworker.glance_data()
        return wrapper
    return decorator