# --------------------------------------------------------
# SpecTNet Metrics
# Copyright (c) 2022 Microsoft
# Adapted from TinyViT
# --------------------------------------------------------

class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self._val = 0
        self._sum = 0
        self._count = 0
        self._avg = None

    def update(self, val, n=1):
        self._val = val
        self._sum += val * n
        self._count += n
        self._avg = None

    @property
    def val(self):
        return self._val

    @property
    def avg(self):
        if self._avg is None:
            self._avg = self._sum / max(1, self._count)
        return self._avg
