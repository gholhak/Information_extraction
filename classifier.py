from sklearn.base import BaseEstimator, TransformerMixin


class MemoryTagger(BaseEstimator, TransformerMixin):

    def fit(self, X, y):
        voc = {}
        self.tags = []
        for x, t in zip(X, y):
            if t not in self.tags:
                self.tags.append(t)
            if x in voc:
                if t in voc[x]:
                    voc[x][t] += 1
                else:
                    voc[x][t] = 1
            else:
                voc[x] = {t: 1}
        self.memory = {}
        for k, d in voc.items():
            self.memory[k] = max(d, key=d.get)

    def predict(self, X, y=None):
        return [self.memory.get(x, 'O') for x in X]
