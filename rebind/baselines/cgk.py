import math
import numpy as np

PAD_VAL = 2

class CGKEnsemble:
    def __init__(self, L: int, m: int = 16, out_scale: float = 6.0, seed: int = 1337):
        self.L = int(L)
        self.m = int(m)
        self.M = int(math.ceil(out_scale * self.L * max(1.0, math.log2(max(2, self.L)))))
        rng = np.random.RandomState(seed)
        self.R = rng.randint(0, 2, size=(self.m, self.M, 2), dtype=np.uint8)

    def embed_batch(self, X: np.ndarray) -> np.ndarray:
        assert X.dtype == np.uint8 and X.ndim == 2
        B, L = X.shape
        Y = np.empty((B, self.m, self.M), dtype=np.uint8)
        for b in range(B):
            xb = X[b]
            for k in range(self.m):
                j = 0
                Rk = self.R[k]
                yk = Y[b, k]
                for t in range(self.M):
                    if j < L:
                        sym = xb[j]
                        yk[t] = sym
                        if Rk[t, sym] == 1:
                            j += 1
                    else:
                        yk[t] = PAD_VAL
        return Y

    @staticmethod
    def _first_pad_idx_row(y: np.ndarray) -> int:
        idx = np.where(y == PAD_VAL)[0]
        return int(idx[0]) if idx.size else y.shape[0]

    def distances_triplet(self, YA: np.ndarray, YP: np.ndarray, YN: np.ndarray,
                          agg: str = "median", mode: str = "ignore_tail"):
        assert YA.dtype == YP.dtype == YN.dtype == np.uint8
        assert YA.shape == YP.shape == YN.shape
        B, m, M = YA.shape

        def pair(Y1, Y2):
            dists = np.empty(m, dtype=float)
            if mode == "ignore_tail":
                for k in range(m):
                    t1 = CGKEnsemble._first_pad_idx_row(Y1[k])
                    t2 = CGKEnsemble._first_pad_idx_row(Y2[k])
                    t = t1 if t1 < t2 else t2
                    dists[k] = 0.0 if t == 0 else float(np.count_nonzero(Y1[k, :t] != Y2[k, :t]))
            else:  # count_all
                for k in range(m):
                    dists[k] = float(np.count_nonzero(Y1[k] != Y2[k]))
            if   agg == "median": return float(np.median(dists))
            elif agg == "min":    return float(np.min(dists))
            else:                 return float(np.mean(dists))

        dap = np.empty(B, dtype=float)
        dan = np.empty(B, dtype=float)
        dpn = np.empty(B, dtype=float)
        for i in range(B):
            dap[i] = pair(YA[i], YP[i])
            dan[i] = pair(YA[i], YN[i])
            dpn[i] = pair(YP[i], YN[i])
        return dap, dan, dpn

    def embed_and_dist_triplet(self, A: np.ndarray, P: np.ndarray, N: np.ndarray,
                               agg: str = "median", mode: str = "ignore_tail"):
        YA = self.embed_batch(A)
        YP = self.embed_batch(P)
        YN = self.embed_batch(N)
        return self.distances_triplet(YA, YP, YN, agg=agg, mode=mode)

def fit_linear_calibrator(raw, true):
    x = np.asarray(raw, float); y = np.asarray(true, float)
    A = np.vstack([x, np.ones_like(x)]).T
    a, b = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(a), float(b)

def apply_linear(x, a, b):
    return a * np.asarray(x, float) + b
