from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class P2Quantile:
    """
    P² quantile estimator (Jain & Chlamtac).
    We store state to dict for persistence.

    This is a pragmatic implementation for latency P99 estimation.
    """
    q: float  # e.g., 0.99

    # marker positions and heights
    n: list[int] | None = None
    np: list[float] | None = None
    dn: list[float] | None = None
    h: list[float] | None = None

    # initial buffer (first 5 samples)
    init: list[float] | None = None

    def __post_init__(self) -> None:
        if self.init is None:
            self.init = []

    def to_state(self) -> dict[str, Any]:
        return {
            "q": self.q,
            "n": self.n,
            "np": self.np,
            "dn": self.dn,
            "h": self.h,
            "init": self.init,
        }

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> "P2Quantile":
        obj = cls(q=float(state.get("q", 0.99)))
        obj.n = state.get("n")
        obj.np = state.get("np")
        obj.dn = state.get("dn")
        obj.h = state.get("h")
        obj.init = state.get("init") or []
        return obj

    def value(self) -> float:
        if self.h is not None and len(self.h) == 5:
            return float(self.h[2])  # middle marker approximates quantile
        if self.init:
            data = sorted(self.init)
            k = int(round((len(data) - 1) * self.q))
            return float(data[max(0, min(len(data) - 1, k))])
        return 0.0

    def update(self, x: float) -> None:
        if self.h is None:
            self.init.append(float(x))
            if len(self.init) < 5:
                return
            self.init.sort()
            self.h = [self.init[0], self.init[1], self.init[2], self.init[3], self.init[4]]
            self.n = [1, 2, 3, 4, 5]
            self.np = [1.0, 1.0 + 2*self.q, 1.0 + 4*self.q, 3.0 + 2*self.q, 5.0]
            self.dn = [0.0, self.q/2.0, self.q, (1.0 + self.q)/2.0, 1.0]
            return

        assert self.h is not None and self.n is not None and self.np is not None and self.dn is not None

        # find k
        if x < self.h[0]:
            self.h[0] = x
            k = 0
        elif x < self.h[1]:
            k = 0
        elif x < self.h[2]:
            k = 1
        elif x < self.h[3]:
            k = 2
        elif x <= self.h[4]:
            k = 3
        else:
            self.h[4] = x
            k = 3

        # increment positions
        for i in range(k + 1, 5):
            self.n[i] += 1
        for i in range(5):
            self.np[i] += self.dn[i]

        # adjust heights of markers 2..4 (index 1..3)
        for i in range(1, 4):
            d = self.np[i] - self.n[i]
            if (d >= 1.0 and self.n[i+1] - self.n[i] > 1) or (d <= -1.0 and self.n[i-1] - self.n[i] < -1):
                di = 1 if d >= 1.0 else -1

                # parabolic prediction
                h_i = self.h[i]
                h_ip1 = self.h[i+1]
                h_im1 = self.h[i-1]
                n_i = self.n[i]
                n_ip1 = self.n[i+1]
                n_im1 = self.n[i-1]

                num = di * (n_i - n_im1 + di) * (h_ip1 - h_i) / (n_ip1 - n_i) + di * (n_ip1 - n_i - di) * (h_i - h_im1) / (n_i - n_im1)
                den = (n_ip1 - n_im1)
                h_new = h_i + num / den if den != 0 else h_i

                # if parabolic goes out of bounds, use linear
                if h_im1 < h_new < h_ip1:
                    self.h[i] = h_new
                else:
                    # linear
                    self.h[i] = h_i + di * (self.h[i + di] - h_i) / (self.n[i + di] - n_i)

                self.n[i] += di
