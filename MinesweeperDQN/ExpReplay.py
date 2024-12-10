import numpy as np


class ExperienceReplay:
    def __init__(self, max_size = 200):
        self.max_size = max_size
        self.memory_st = []
        self.memory_qt = []

    def memorize(self, state, Qtar):
        self.memory_st.append(state)
        self.memory_qt.append(Qtar)

        if len(self.memory_st) > self.max_size:
            del self.memory_st[0]
            del self.memory_qt[0]

    def get_batch(self, bsize):
        if bsize > len(self.memory_qt):
            return self.memory_st, self.memory_qt, len(self.memory_qt)

        indx = np.random.randint(0, len(self.memory_qt), bsize)
        x = []
        y = []
        for i in indx:
            x.append(self.memory_st[i])
            y.append(self.memory_qt[i])
        return x, y, bsize
