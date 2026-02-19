import time
import logging

log = logging.getLogger("timing")

class T:
    def __init__(self, label: str):
        self.label = label
        self.t0 = time.perf_counter()

    def mark(self, step: str):
        dt = (time.perf_counter() - self.t0) * 1000
        log.info("[%s] %s: %.1f ms", self.label, step, dt)
        self.t0 = time.perf_counter()
