import numpy as np
from dataclasses import dataclass
from scipy.stats import invgauss, uniform


""" Constants """
EPS = np.finfo(float).tiny   
THRESHOLD = 1.0         # fixed: not identifiable together with sigma
MIN_C = 1e-12           # stands in for zero contrast inside the drift, which keeps it positive and differentiable


""" Data classes """
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
    c50: float | None = None 
    # c50: contrast of half-max drift, saturates at gain.
    # if none, power law with no saturation

@dataclass
class LapseParams:
    prob: float                 # catch false alarm rate
    dist: str = "uniform"       # lapse dist [invgauss, uniform]
    # lapse dist parameters
    tau: float | None = None    
    shape: float | None = None  

# TODO: figure out best to to unify with response times window [150-1000 ms]
@dataclass
class TaskParams:            
    floor: float = 0.100     # earliest hit rt
    deadline: float = 0.870  # latest hit rt 


""" Model classes """
class GatedDiffusion:
    """ Simple gated-model: no lapse dist """
    def __init__(self, gates, diffusion, task=None):
        self.gates = gates                  # dict: stim_type -> GateParams
        self.diffusion = diffusion          # shared DiffusionParams
        self.task = task or TaskParams()    # fixed task constants

    # --- Shared by all variants ---
    # q(c): probability that usable sensory evidence is available
    def gate_prob(self, stim, contrast):
        q, known = 0.0, False
        for name, g in self.gates.items():
            weibull = 1 - np.exp(-(contrast / g.scale) ** g.shape)
            q = np.where(stim == name, g.ceiling * weibull, q)
            known = known | (stim == name)
        if not np.all(known):
            raise ValueError(f"unknown stim_type: {set(np.unique(stim)) - set(self.gates)}")
        return q

    # contrast-to-drift function, same for both SFTF conditions
    def drift(self, contrast):
        #   saturating:  gain * c**gamma / (c**gamma + c50**gamma)     (Naka-Rushton, when c50 is set)
        #   power law:   gain * c**gamma            
        d = self.diffusion
        signal = np.maximum(contrast, MIN_C) ** d.gamma
        if d.c50 is not None:
            signal = signal / (signal + d.c50 ** d.gamma)
        return d.gain * signal

    # inverse-Gaussian first-passage-time distribution of the diffusion: f_E, F_E
    def _fpt(self, contrast):
        mean = THRESHOLD / self.drift(contrast)
        shape = (THRESHOLD / self.diffusion.sigma) ** 2
        return invgauss(mu=mean / shape, scale=shape)

    # gate probability q, the evidence distribution E, and p(no response before the floor).
    # Only the evidence process can respond that early, and only if it starts before the floor.
    def _gate(self, stim, contrast):
        q, E = self.gate_prob(stim, contrast), self._fpt(contrast)
        return q, E, 1 - q * E.cdf(self.task.floor - self.diffusion.t0)

    # gate (open / closed) and evidence response time (from stimulus onset) for n repeats of each condition
    def _sample(self, stim, contrast, n):
        q = self.gate_prob(stim, contrast)
        size = (n, *np.shape(q))
        gate = np.random.random(size) < q
        return gate, self.diffusion.t0 + self._fpt(contrast).rvs(size)

    # --- Changed by variants ---
    # joint density p(hit, rt | stim, contrast)
    def rt_pdf(self, stim, contrast, rt):
        q, E, not_early = self._gate(stim, contrast)
        return q * E.pdf(rt - self.diffusion.t0) / not_early

    # p(response by rt | stim, contrast)
    def rt_cdf(self, stim, contrast, rt):
        q, E, not_early = self._gate(stim, contrast)
        return q * (E.cdf(rt - self.diffusion.t0) - E.cdf(self.task.floor - self.diffusion.t0)) / not_early

    # response time from stimulus onset for n trials per condition (inf: no response)
    def _response_time(self, stim, contrast, n):
        gate, t_evidence = self._sample(stim, contrast, n)
        return np.where(gate, t_evidence, np.inf)

    # --- Derrived by subclasses ---
    # generate n trials for each stimulus condition, returns (hit, rt)
    # Trials with a response before the floor are early trials: they are redrawn, as in the data
    def simulate(self, stim, contrast, n=1):
        rt = self._response_time(stim, contrast, n)
        while np.any(early := rt < self.task.floor):
            rt = np.where(early, self._response_time(stim, contrast, n), rt)
        hit = rt < self.task.deadline
        return hit, np.where(hit, rt, np.nan)

    # psychometric: p(hit | stim, contrast)
    def hit_prob(self, stim, contrast):
        return self.rt_cdf(stim, contrast, self.task.deadline)

    # chronometric: quantiles p of rt on hit trials, shape (len(p), n_conditions); p = [0.5] is the median
    # found by inverting rt_cdf on a time grid, for all conditions at once (nan where a hit is impossible)
    def rt_quantile(self, stim, contrast, p, n_grid=400):
        rt = np.linspace(self.task.floor, self.task.deadline, n_grid)
        cdf = self.rt_cdf(stim, contrast, rt[:, None])      # (n_grid, n_conditions)
        return np.array([[np.interp(pi * c[-1], c, rt) if c[-1] > 0 else np.nan for c in cdf.T] for pi in p])

    # log p(hit, rt | params), summed over trials; rt is ignored on misses.
    # Every hit is scored on its reaction time, catch trials included. A catch-trial hit is a pure lapse,
    # so its term is log(lapse prob) + log(lapse timing density at rt). The timing has no fitted
    # parameters (flat, or an inverse Gaussian with fixed tau and shape), so catch reaction times add a
    # constant and cannot move any estimate; they inform the fit only through the lapse prob.
    def log_likelihood(self, stim, contrast, hit, rt):
        log_hit = np.log(np.maximum(self.rt_pdf(stim, contrast, rt), EPS))
        log_miss = np.log(np.maximum(1 - self.hit_prob(stim, contrast), EPS))
        return np.sum(np.where(hit, log_hit, log_miss))


class Lapse:
    """ Distribution of spontaneous activity """

    def __init__(self, prob, timing, window):
        self.prob, self.timing, self.window = prob, timing, window
        self.before = timing.cdf(0)                         # timing before the window
        self.in_window = timing.cdf(window) - self.before   # timing inside window

    def pdf(self, t):
        return np.where((t >= 0) & (t <= self.window), self.prob * self.timing.pdf(t) / self.in_window, 0.0)

    def cdf(self, t):
        return self.prob * np.clip((self.timing.cdf(t) - self.before) / self.in_window, 0.0, 1.0)

    def sf(self, t):
        return 1 - self.cdf(t)

    def rvs(self, size):
        t = self.timing.ppf(self.before + np.random.random(size) * self.in_window)
        return np.where(np.random.random(size) < self.prob, t, np.inf)


# lapse-time distribution f_L, F_L over time since the floor; every option gives cdf(window) = prob
def lapse_dist(lapse, floor, window):
    if lapse.dist == "uniform":
        # flat over the window
        return Lapse(lapse.prob, uniform(scale=window), window)
    if lapse.dist == "invgauss":
        # a second accumulation-to-threshold process, timed from stimulus onset (loc shifts it to the floor)
        return Lapse(lapse.prob, invgauss(lapse.tau / lapse.shape, loc=-floor, scale=lapse.shape), window)
    raise ValueError(f"unknown lapse dist: {lapse.dist}")


class GatedDiffusionMixture(GatedDiffusion):
    """ Each trial is drawn from either the evidence process or the lapse process. """

    def __init__(self, gates, diffusion, lapse, task=None):
        super().__init__(gates, diffusion, task)
        self.lapse = lapse                  # LapseParams

    def _lapse(self):
        return lapse_dist(self.lapse, self.task.floor, self.task.deadline - self.task.floor)

    # t_E and t_L are the time since the start of the evidence and of the lapse process
    def rt_pdf(self, stim, contrast, rt):
        (q, E, not_early), L = self._gate(stim, contrast), self._lapse()
        t_E, t_L = rt - self.diffusion.t0, rt - self.task.floor
        return (q * E.pdf(t_E) + (1 - q) * L.pdf(t_L)) / not_early

    def rt_cdf(self, stim, contrast, rt):
        (q, E, not_early), L = self._gate(stim, contrast), self._lapse()
        t_E, t_L = rt - self.diffusion.t0, rt - self.task.floor
        return (q * (E.cdf(t_E) - E.cdf(self.task.floor - self.diffusion.t0)) + (1 - q) * L.cdf(t_L)) / not_early

    def _response_time(self, stim, contrast, n):
        gate, t_evidence = self._sample(stim, contrast, n)
        t_lapse = self.task.floor + self._lapse().rvs(gate.shape)
        return np.where(gate, t_evidence, t_lapse)


class GatedDiffusionRace(GatedDiffusion):
    """ Evidence and lapse processes race; the first to finish sets the response. """

    def __init__(self, gates, diffusion, lapse, task=None):
        super().__init__(gates, diffusion, task)
        self.lapse = lapse                  # LapseParams

    def _lapse(self):
        return lapse_dist(self.lapse, self.task.floor, self.task.deadline - self.task.floor)

    # sf is the survival function 1 - cdf: the process has not finished yet
    def rt_pdf(self, stim, contrast, rt):
        (q, E, not_early), L = self._gate(stim, contrast), self._lapse()
        t_E, t_L = rt - self.diffusion.t0, rt - self.task.floor
        return (q * (E.pdf(t_E) * L.sf(t_L) + L.pdf(t_L) * E.sf(t_E)) + (1 - q) * L.pdf(t_L)) / not_early

    def rt_cdf(self, stim, contrast, rt):
        (q, E, not_early), L = self._gate(stim, contrast), self._lapse()
        t_E, t_L = rt - self.diffusion.t0, rt - self.task.floor
        no_response_at_floor = E.sf(self.task.floor - self.diffusion.t0)
        return (q * (no_response_at_floor - E.sf(t_E) * L.sf(t_L)) + (1 - q) * L.cdf(t_L)) / not_early

    def _response_time(self, stim, contrast, n):
        gate, t_evidence = self._sample(stim, contrast, n)
        t_lapse = self.task.floor + self._lapse().rvs(gate.shape)
        return np.where(gate, np.minimum(t_evidence, t_lapse), t_lapse)
