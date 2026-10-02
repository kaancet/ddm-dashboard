from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.stats import invgauss, uniform
from scipy.stats.distributions import rv_frozen

Array = np.ndarray

# --- constants ---

EPS = np.finfo(float).tiny   # floor for probabilities before taking the log
THRESHOLD = 1.0              # fixed: not identifiable together with sigma
MIN_C = 1e-12                # stands in for zero contrast in the drift: keeps it positive and differentiable


# --- parameters ---

@dataclass
class GateParams:
    scale: float
    shape: float
    ceiling: float = 1.0


@dataclass
class DiffusionParams:
    gain: float
    gamma: float
    sigma: float
    t0: float = 0.100
    c50: float | None = None    # contrast of half-max drift, saturates at gain. None: power law, no saturation


@dataclass
class LapseParams:
    prob: float                 # catch false alarm rate
    dist: Literal["uniform", "invgauss"] = "uniform"
    # invgauss parameters
    mu: float | None = None     # mean (s)
    lam: float | None = None    # shape


# TODO: figure out best way to unify with response times window [150-1000 ms]
@dataclass
class TaskParams:
    floor: float = 0.100     # earliest hit rt
    deadline: float = 0.870  # latest hit rt


# --- models ---

class GatedDiffusion:
    """ Simple gated-model: no lapse dist """

    def __init__(self, gates: dict[str, GateParams], diffusion: DiffusionParams, task: TaskParams | None = None):
        self.gates = gates                  # one per stimulus
        self.diffusion = diffusion          # shared by all stimuli
        self.task = task or TaskParams()    # fixed task constants

    # --- Shared by all variants ---

    def gate_prob(self, stim: Array | str, contrast: Array) -> Array:
        """ q(c): probability that usable sensory evidence is available """
        q, known = 0.0, False
        for name, g in self.gates.items():
            weibull = 1 - np.exp(-(contrast / g.scale) ** g.shape)
            q = np.where(stim == name, g.ceiling * weibull, q)
            known = known | (stim == name)
        if not np.all(known):
            raise ValueError(f"unknown stim_type: {set(np.unique(stim)) - set(self.gates)}")
        return q

    def drift(self, contrast: Array) -> Array:
        """ Contrast-to-drift function, same for both SFTF conditions """
        #   saturating:  gain * c**gamma / (c**gamma + c50**gamma)     (Naka-Rushton, when c50 is set)
        #   power law:   gain * c**gamma
        d = self.diffusion
        signal = np.maximum(contrast, MIN_C) ** d.gamma
        if d.c50 is not None:
            signal = signal / (signal + d.c50 ** d.gamma)
        return d.gain * signal

    def _fpt(self, contrast: Array) -> rv_frozen:
        """ Inverse-Gaussian first-passage-time distribution of the diffusion: f_E, F_E """
        mu = THRESHOLD / self.drift(contrast)               # mean
        lam = (THRESHOLD / self.diffusion.sigma) ** 2       # shape
        return invgauss(mu=mu / lam, scale=lam)

    def _gate_and_evidence(self, stim: Array | str, contrast: Array) -> tuple[Array, rv_frozen, Array]:
        """ Gate probability q, evidence distribution E, and p(no response before the floor) """
        # only the evidence process can respond that early, and only if it starts before the floor
        q, E = self.gate_prob(stim, contrast), self._fpt(contrast)
        return q, E, 1 - q * E.cdf(self.task.floor - self.diffusion.t0)

    def _sample(self, stim: Array | str, contrast: Array, n: int) -> tuple[Array, Array]:
        """ Gate (open / closed) and evidence response time from stimulus onset, n repeats per condition """
        q = self.gate_prob(stim, contrast)
        size = (n, *np.shape(q))
        gate = np.random.random(size) < q
        return gate, self.diffusion.t0 + self._fpt(contrast).rvs(size)

    # --- Changed by variants ---

    def rt_pdf(self, stim: Array | str, contrast: Array, rt: Array) -> Array:
        """ Joint density p(hit, rt | stim, contrast) """
        q, E, not_early = self._gate_and_evidence(stim, contrast)
        return q * E.pdf(rt - self.diffusion.t0) / not_early

    def rt_cdf(self, stim: Array | str, contrast: Array, rt: Array) -> Array:
        """ p(response by rt | stim, contrast) """
        q, E, not_early = self._gate_and_evidence(stim, contrast)
        return q * (E.cdf(rt - self.diffusion.t0) - E.cdf(self.task.floor - self.diffusion.t0)) / not_early

    def _response_time(self, stim: Array | str, contrast: Array, n: int) -> Array:
        """ Response time from stimulus onset, n trials per condition (inf: no response) """
        gate, t_evidence = self._sample(stim, contrast, n)
        return np.where(gate, t_evidence, np.inf)

    # --- Derived, same for all variants ---

    def simulate(self, stim: Array, contrast: Array, n: int = 1) -> tuple[Array, Array]:
        """ Generate n trials per condition, returns (hit, rt) """
        # trials with a response before the floor are early trials: they are redrawn, as in the data
        rt = self._response_time(stim, contrast, n)
        while np.any(early := rt < self.task.floor):
            rt = np.where(early, self._response_time(stim, contrast, n), rt)
        hit = rt < self.task.deadline
        return hit, np.where(hit, rt, np.nan)

    def hit_prob(self, stim: Array | str, contrast: Array) -> Array:
        """ Psychometric: p(hit | stim, contrast) """
        return self.rt_cdf(stim, contrast, self.task.deadline)

    def log_likelihood(self, stim: Array, contrast: Array, hit: Array, rt: Array) -> float:
        """ log p(hit, rt | params), summed over trials; rt is ignored on misses """
        log_hit = np.log(np.maximum(self.rt_pdf(stim, contrast, rt), EPS))
        log_miss = np.log(np.maximum(1 - self.hit_prob(stim, contrast), EPS))
        return np.sum(np.where(hit, log_hit, log_miss))


class LapseDist:
    """ Distribution of spontaneous activity """

    def __init__(self, prob: float, timing: rv_frozen, window: float):
        self.prob, self.timing, self.window = prob, timing, window
        self.before = timing.cdf(0)                         # timing before the window
        self.in_window = timing.cdf(window) - self.before   # timing inside window

    def pdf(self, t: Array) -> Array:
        return np.where((t >= 0) & (t <= self.window), self.prob * self.timing.pdf(t) / self.in_window, 0.0)

    def cdf(self, t: Array) -> Array:
        return self.prob * np.clip((self.timing.cdf(t) - self.before) / self.in_window, 0.0, 1.0)

    def sf(self, t: Array) -> Array:
        return 1 - self.cdf(t)

    def rvs(self, size: tuple[int, ...]) -> Array:
        t = self.timing.ppf(self.before + np.random.random(size) * self.in_window)
        return np.where(np.random.random(size) < self.prob, t, np.inf)


def lapse_dist(lapse: LapseParams, floor: float, window: float) -> LapseDist:
    """ Lapse-time distribution f_L, F_L over time since the floor; every option gives cdf(window) = prob """
    if lapse.dist == "uniform":
        # flat over the window
        return LapseDist(lapse.prob, uniform(scale=window), window)
    if lapse.dist == "invgauss":
        # a second accumulation-to-threshold process, timed from stimulus onset (loc shifts it to the floor)
        return LapseDist(lapse.prob, invgauss(lapse.mu / lapse.lam, loc=-floor, scale=lapse.lam), window)
    raise ValueError(f"unknown lapse dist: {lapse.dist}")


class GatedDiffusionMixture(GatedDiffusion):
    """ Each trial is drawn from either the evidence process or the lapse process. """

    def __init__(self, gates: dict[str, GateParams], diffusion: DiffusionParams, lapse: LapseParams,
                 task: TaskParams | None = None):
        super().__init__(gates, diffusion, task)
        self.lapse = lapse

    def _lapse_dist(self) -> LapseDist:
        return lapse_dist(self.lapse, self.task.floor, self.task.deadline - self.task.floor)

    # t_E and t_L are the time since the start of the evidence and of the lapse process
    def rt_pdf(self, stim: Array | str, contrast: Array, rt: Array) -> Array:
        q, E, not_early = self._gate_and_evidence(stim, contrast)
        L = self._lapse_dist()
        t_E, t_L = rt - self.diffusion.t0, rt - self.task.floor
        return (q * E.pdf(t_E) + (1 - q) * L.pdf(t_L)) / not_early

    def rt_cdf(self, stim: Array | str, contrast: Array, rt: Array) -> Array:
        q, E, not_early = self._gate_and_evidence(stim, contrast)
        L = self._lapse_dist()
        t_E, t_L = rt - self.diffusion.t0, rt - self.task.floor
        return (q * (E.cdf(t_E) - E.cdf(self.task.floor - self.diffusion.t0)) + (1 - q) * L.cdf(t_L)) / not_early

    def _response_time(self, stim: Array | str, contrast: Array, n: int) -> Array:
        gate, t_evidence = self._sample(stim, contrast, n)
        t_lapse = self.task.floor + self._lapse_dist().rvs(gate.shape)
        return np.where(gate, t_evidence, t_lapse)


class GatedDiffusionRace(GatedDiffusion):
    """ Evidence and lapse processes race; the first to finish sets the response. """

    def __init__(self, gates: dict[str, GateParams], diffusion: DiffusionParams, lapse: LapseParams,
                 task: TaskParams | None = None):
        super().__init__(gates, diffusion, task)
        self.lapse = lapse

    def _lapse_dist(self) -> LapseDist:
        return lapse_dist(self.lapse, self.task.floor, self.task.deadline - self.task.floor)

    # sf is the survival function 1 - cdf: the process has not finished yet
    def rt_pdf(self, stim: Array | str, contrast: Array, rt: Array) -> Array:
        q, E, not_early = self._gate_and_evidence(stim, contrast)
        L = self._lapse_dist()
        t_E, t_L = rt - self.diffusion.t0, rt - self.task.floor
        return (q * (E.pdf(t_E) * L.sf(t_L) + L.pdf(t_L) * E.sf(t_E)) + (1 - q) * L.pdf(t_L)) / not_early

    def rt_cdf(self, stim: Array | str, contrast: Array, rt: Array) -> Array:
        q, E, not_early = self._gate_and_evidence(stim, contrast)
        L = self._lapse_dist()
        t_E, t_L = rt - self.diffusion.t0, rt - self.task.floor
        no_response_at_floor = E.sf(self.task.floor - self.diffusion.t0)
        return (q * (no_response_at_floor - E.sf(t_E) * L.sf(t_L)) + (1 - q) * L.cdf(t_L)) / not_early

    def _response_time(self, stim: Array | str, contrast: Array, n: int) -> Array:
        gate, t_evidence = self._sample(stim, contrast, n)
        t_lapse = self.task.floor + self._lapse_dist().rvs(gate.shape)
        return np.where(gate, np.minimum(t_evidence, t_lapse), t_lapse)
