class EarlyStopping:
    def __init__(self, patience: int, mode: str = 'min'):
        if mode not in ('min', 'max'):
            raise ValueError("mode must be 'min' or 'max'")

        self.patience = patience
        self.mode = mode
        self.best = None
        self.bad_steps = 0

    def _is_improvement(self, metric: float) -> bool:
        if self.best is None:
            return True
        if self.mode == 'min':
            return metric < self.best
        else:  # mode == 'max'
            return metric > self.best

    def step(self, metric: float) -> bool:
        if self._is_improvement(metric):
            self.best = metric
            self.bad_steps = 0
            return False
        else:
            self.bad_steps += 1
            return self.bad_steps >= self.patience