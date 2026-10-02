import math
from typing import Callable, Literal

import numpy as np
import polars as pl
import torch
from sbi.inference.posteriors import MCMCPosterior
from sbi.utils import BoxUniform, mcmc_transform
from scipy.optimize import minimize
from torch import Tensor
from torch.special import log_ndtr, ndtr

from .model import (EPS, MIN_C, THRESHOLD, DiffusionParams, GateParams, GatedDiffusion, GatedDiffusionMixture,
                    GatedDiffusionRace, LapseParams, TaskParams)

Params = dict[str, float]       # parameter name -> value


# --- parameters ---

def param_names(stims: list[str], saturating: bool = True) -> list[str]:
    """ Parameter names, in the order of the flat vectors used in the fits. The only place the order is defined

    saturating: the drift saturates with contrast and has a c50; otherwise it is a power law
    """
    names = ["gain", "gamma", "sigma", "ceiling"] + [f"{kind}_{s}" for s in stims for kind in ("scale", "shape")]
    names.append("t0")
    if saturating:
        names.append("c50")
    return names


# uniform prior range of each parameter. scale and shape apply to every stimulus
BOUNDS = {"gain": (1e-3, 100.), "gamma": (1e-3, 3.), "sigma": (1e-3, 5.), "ceiling": (0.5, 1.0),
          "scale": (1e-3, 1.0), "shape": (0.1, 10.),
          "t0": (0.0, 0.3),         # s
          "c50": (1e-3, 1.0)}       # contrast


def param_bounds(names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """ (low, high) for the given parameter names """
    kinds = [name if name in BOUNDS else name.split("_")[0] for name in names]
    bounds = [BOUNDS[kind] for kind in kinds]
    low, high = zip(*bounds)
    return np.array(low), np.array(high)


def build_model(params: Params, stims: list[str], variant: type[GatedDiffusion] = GatedDiffusion,
                task: TaskParams | None = None, lapse: LapseParams | None = None) -> GatedDiffusion:
    """ Parameters -> model of the given variant. Without a c50 the drift is a power law

    lapse: needed by the lapse variants
    """
    expected = param_names(stims, saturating="c50" in params)
    if set(params) != set(expected):
        raise ValueError(f"missing parameters: {sorted(set(expected) - set(params))}, "
                         f"unknown parameters: {sorted(set(params) - set(expected))}")
    diffusion = DiffusionParams(params["gain"], params["gamma"], params["sigma"], params["t0"], params.get("c50"))
    # the ceiling is shared
    gates = {s: GateParams(params[f"scale_{s}"], params[f"shape_{s}"], params["ceiling"]) for s in stims}
    if variant is GatedDiffusion:
        return variant(gates, diffusion, task)
    return variant(gates, diffusion, lapse, task)


def _check_fixed(fixed: Params | None, names: list[str]) -> Params:
    """ The fixed parameters, checked against the parameter names """
    fixed = dict(fixed or {})
    if unknown := set(fixed) - set(names):
        raise ValueError(f"unknown fixed parameters: {sorted(unknown)}, expected names from {list(names)}")
    return fixed


# --- trial-wise log-likelihood in PyTorch ---
# The maths of `log_likelihood` in model.py, with one row of parameters per MCMC chain

MIN_T = 1e-9    # stands in for times at or before the start of a process (density and cdf are 0 there)


def _fpt(t: Tensor, mu: Tensor, lam: Tensor) -> tuple[Tensor, Tensor]:
    """ Inverse-Gaussian first-passage pdf and cdf at time t, for mean mu and shape lam """
    t = t.clamp(min=MIN_T)
    pdf = torch.sqrt(lam / (2 * math.pi * t ** 3)) * torch.exp(-lam * (t - mu) ** 2 / (2 * mu ** 2 * t))
    a = torch.sqrt(lam / t)
    # second term is exp(2 lam / mu) * ndtr(...), kept in log space so it cannot overflow
    cdf = ndtr(a * (t / mu - 1)) + torch.exp(2 * lam / mu + log_ndtr(-a * (t / mu + 1)))
    return pdf, cdf


def _lapse_pdf_sf(rt: Tensor, lapse: LapseParams, floor: float, window: float) -> tuple[Tensor, Tensor]:
    """ Lapse density f_L and survival S_L at reaction time rt """
    p = lapse.prob
    if lapse.dist == "uniform":                     # flat density p / window over the window
        return torch.full_like(rt, p / window), 1 - p * (rt - floor) / window
    if lapse.dist == "invgauss":                    # timed from stimulus onset, restricted to the window
        mu, lam = torch.tensor(lapse.mu, dtype=rt.dtype), torch.tensor(lapse.lam, dtype=rt.dtype)
        window_edges = torch.tensor([floor, floor + window], dtype=rt.dtype)
        start, end = _fpt(window_edges, mu, lam)[1]    # cdf at the edges
        in_window = end - start
        timing, done = _fpt(rt, mu, lam)
        density = p * timing / in_window
        survival = 1 - p * (done - start) / in_window
        return density, survival
    raise ValueError(f"unknown lapse dist: {lapse.dist}")


def log_likelihood(theta: Tensor, names: list[str], stim_index: Tensor, contrast: Tensor, hit: Tensor, rt: Tensor,
                   variant: type[GatedDiffusion] = GatedDiffusion, task: TaskParams | None = None,
                   lapse: LapseParams | None = None) -> Tensor:
    """ Log-likelihood summed over trials, shape (n_chains,)

    theta: (n_chains, n_params), one column per entry of names. Without a c50 column the drift is a power law
    stim_index: position of each trial's stimulus in the stims list given to `param_names`
    lapse: needed by the lapse variants
    """
    task = task or TaskParams()
    floor, window = task.floor, task.deadline - task.floor

    theta = theta[:, :, None]                       # (n_chains, n_params, 1): broadcasts over trials
    column = {name: theta[:, i] for i, name in enumerate(names)}
    gain, gamma, sigma, ceiling, t0 = (column[name] for name in ("gain", "gamma", "sigma", "ceiling", "t0"))
    # gate parameters of each trial's stimulus, (n_chains, n_trials). The stimuli are in the order of names
    scale = torch.cat([column[name] for name in names if name.startswith("scale_")], dim=1)[:, stim_index]
    shape = torch.cat([column[name] for name in names if name.startswith("shape_")], dim=1)[:, stim_index]

    c = contrast.clamp(min=MIN_C)
    rt = torch.where(hit, rt, torch.full_like(rt, floor + window / 2))    # placeholder time on misses

    # gate, and the evidence process at the trial's time, at the floor and at the deadline
    q = ceiling * (1 - torch.exp(-(c / scale) ** shape)) * (contrast > 0)      # exactly 0 on catch trials
    signal = c ** gamma
    if "c50" in column:
        c50 = column["c50"]
        signal = signal / (signal + c50 ** gamma)
    mu, lam = THRESHOLD / (gain * signal), (THRESHOLD / sigma) ** 2
    f_E, F_E = _fpt(rt - t0, mu, lam)
    F_E_floor = _fpt(floor - t0 + torch.zeros_like(mu), mu, lam)[1]
    F_E_end = _fpt(floor + window - t0 + torch.zeros_like(mu), mu, lam)[1]

    # p(no response before the floor): only the evidence process can respond that early
    not_early = 1 - q * F_E_floor

    if variant is GatedDiffusion:
        pdf, miss = q * f_E, q * (1 - F_E_end) + (1 - q)
    else:
        p = lapse.prob
        f_L, S_L = _lapse_pdf_sf(rt, lapse, floor, window)
        if variant is GatedDiffusionMixture:
            pdf = q * f_E + (1 - q) * f_L
            miss = q * (1 - F_E_end) + (1 - q) * (1 - p)
        elif variant is GatedDiffusionRace:
            pdf = q * (f_E * S_L + f_L * (1 - F_E)) + (1 - q) * f_L
            miss = (q * (1 - F_E_end) + (1 - q)) * (1 - p)
    pdf, miss = pdf / not_early, miss / not_early

    log_hit = torch.log(pdf.clamp(min=EPS))
    log_miss = torch.log(miss.clamp(min=EPS))
    return torch.where(hit, log_hit, log_miss).sum(dim=1)


# --- fits ---

def _laplace(potential: Callable[[Tensor, None], Tensor], prior: BoxUniform, num_samples: int,
             num_starts: int = 500) -> Tensor:
    """ Gaussian approximation to the posterior at its mode (Laplace) """
    # built in sbi's unconstrained parameter space, so samples always respect the prior bounds
    transform = mcmc_transform(prior)                   # theta -> unconstrained z

    def neg_log_posterior(z):                           # z: (n, n_params), density over z
        theta = transform.inv(z)
        return -(potential(theta, None) - transform.log_abs_det_jacobian(theta, z))

    # start from the best of many prior draws, then climb to the mode
    z0 = transform(prior.sample((num_starts,)))
    z = z0[neg_log_posterior(z0).argmin()].clone().requires_grad_(True)
    optimiser = torch.optim.LBFGS([z], max_iter=500, line_search_fn="strong_wolfe")

    def closure():
        optimiser.zero_grad()
        loss = neg_log_posterior(z[None])[0]
        loss.backward()
        return loss

    optimiser.step(closure)
    z = z.detach()

    # curvature at the mode gives the covariance
    hessian = torch.autograd.functional.hessian(lambda z: neg_log_posterior(z[None])[0], z)
    gaussian = torch.distributions.MultivariateNormal(z, precision_matrix=hessian)
    return transform.inv(gaussian.sample((num_samples,)))


def fit_trials(stim: np.ndarray, contrast: np.ndarray, hit: np.ndarray, rt: np.ndarray,
               variant: type[GatedDiffusion] = GatedDiffusion, task: TaskParams | None = None,
               method: Literal["nuts", "laplace"] = "nuts", lapse: LapseParams | None = None,
               saturating: bool = True, fixed: Params | None = None,
               num_samples: int = 2000, num_chains: int = 4, warmup_steps: int = 500) -> pl.DataFrame:
    """ Posterior samples: one row per draw, one column per parameter. Uniform prior over BOUNDS

    stim, contrast, hit, rt: one entry per trial (rt is ignored on misses)
    method: "nuts" (sbi's NUTS sampler, minutes) or "laplace" (Gaussian at the mode, seconds)
    lapse: needed by the lapse variants. Measured, not sampled
    fixed: parameters held fixed, returned as constant columns
    """
    stims = list(np.unique(stim))
    names = param_names(stims, saturating)
    fixed = _check_fixed(fixed, names)
    stim_index = torch.as_tensor(np.searchsorted(stims, stim))
    contrast, hit, rt = (torch.tensor(np.asarray(x)) for x in (contrast, hit, rt))     # copies: inputs may be read-only
    low, high = (torch.tensor(bound) for bound in param_bounds(names))
    sampled = [i for i, name in enumerate(names) if name not in fixed]
    prior = BoxUniform(low[sampled], high[sampled])

    # the full parameter vector: the sampled parameters, with the fixed ones put in their places
    def complete(theta):
        full = torch.empty((len(theta), len(names)), dtype=theta.dtype)
        full[:, sampled] = theta
        for name, value in fixed.items():
            full[:, names.index(name)] = value
        return full

    # unnormalised log posterior, for one row of theta per chain
    def potential(theta, x_o):
        theta = torch.atleast_2d(theta).double()
        log_lik = log_likelihood(complete(theta), names, stim_index, contrast, hit, rt,
                                 variant, task, lapse)
        return prior.log_prob(theta) + log_lik

    if method == "laplace":
        samples = _laplace(potential, prior, num_samples)
    elif method == "nuts":
        # sbi is used only for its sampler; the potential above replaces a learned one
        posterior = MCMCPosterior(potential, proposal=prior, theta_transform=mcmc_transform(prior),
                                  method="nuts_pyro", num_chains=1, warmup_steps=warmup_steps,
                                  thin=1, init_strategy="sir")
        # chains run one after another, each with its own start and warm-up
        # (Pyro's parallel chains need a picklable potential). x is unused: the data are in `potential`
        chains = [posterior.sample((num_samples // num_chains,), x=torch.zeros(1, 1)) for _ in range(num_chains)]
        samples = torch.cat(chains)
    else:
        raise ValueError(f"unknown method: {method}")
    return pl.DataFrame(complete(samples).numpy(), schema=names, orient="row")


def map_estimate(posterior: pl.DataFrame, fitted: list[str], build: Callable[[Params], GatedDiffusion],
                 stim: np.ndarray, contrast: np.ndarray, hit: np.ndarray, rt: np.ndarray) -> tuple[Params, float]:
    """ Parameters of highest likelihood (the MAP, as the prior is uniform) and their log-likelihood

    Climbs from the posterior mean over the fitted parameters (names in `fitted`)
    """
    start = posterior.mean().row(0, named=True)

    def neg_log_lik(x):
        params = start | dict(zip(fitted, x))
        model = build(params)
        value = -model.log_likelihood(stim, contrast, hit, rt)
        return value if np.isfinite(value) else np.inf

    result = minimize(neg_log_lik, [start[name] for name in fitted], method="Nelder-Mead",
                      options=dict(xatol=1e-7, fatol=1e-7, maxiter=5000))
    map_params = start | {name: float(value) for name, value in zip(fitted, result.x)}
    return map_params, -result.fun
