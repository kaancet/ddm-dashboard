""" Fitting

param_names:    the parameters and their order
build_model:    parameters -> model
log_likelihood: trial-wise log-likelihood in PyTorch, the one used for fitting
fit_trials:     Bayesian fit of the trial-wise likelihood (posterior samples)
map_estimate:   parameter vector of highest likelihood, found from the posterior mean
"""
import math

import numpy as np
import polars as pl
import torch
from sbi.inference.posteriors import MCMCPosterior
from sbi.utils import BoxUniform, mcmc_transform
from scipy.optimize import minimize
from torch.special import log_ndtr, ndtr

from .model import (MIN_C, THRESHOLD, DiffusionParams, GateParams, GatedDiffusion, GatedDiffusionMixture,
                    GatedDiffusionRace, LapseParams, TaskParams)


# --- parameters ---

# The parameters, in the order of the flat vectors used inside the fits:
#   gain, gamma, sigma, ceiling, then scale_<stim> and shape_<stim> for each stimulus,
#   lapse_prob (lapse variants only), t0, c50 (saturating drift only)
# This is the only place the order is defined: everywhere else a parameter is found by its name.
# A parameter that is not fitted is held at a value with the `fixed` argument of the fit functions,
# e.g. t0 at the floor, or lapse_prob at the catch-trial hit rate.
# saturating: if True (default), the drift saturates with contrast and has a c50; if False, it is a power law
def param_names(stims, model=GatedDiffusion, saturating=True):
    names = ["gain", "gamma", "sigma", "ceiling"] + [f"{p}_{s}" for s in stims for p in ("scale", "shape")]
    if model is not GatedDiffusion:
        names.append("lapse_prob")
    names.append("t0")
    if saturating:
        names.append("c50")
    return names


# allowed range of each parameter: the uniform prior of the Bayesian fit.
# scale and shape apply to every stimulus
BOUNDS = {"gain": (1e-3, 100.), "gamma": (1e-3, 3.), "sigma": (1e-3, 5.), "ceiling": (0.5, 1.0),
          "scale": (1e-3, 1.0), "shape": (0.1, 10.), "lapse_prob": (1e-3, 0.999),
          "t0": (0.0, 0.3),         # s
          "c50": (1e-3, 1.0)}       # contrast


# (low, high) for the given parameter names
def param_bounds(names):
    low, high = zip(*(BOUNDS[name] if name in BOUNDS else BOUNDS[name.split("_")[0]] for name in names))
    return np.array(low), np.array(high)


# parameters -> model
# theta: dict parameter name -> value, with the names of `param_names`. Without a c50 the drift is a power law
# lapse_dist: lapse-time shape, "uniform" (flat over the window) or "invgauss" (inverse Gaussian timed
#             from stimulus onset, with fixed mean time lapse_tau (s) and shape lapse_shape)
def build_model(theta, stims, model=GatedDiffusion, task=None, lapse_dist="uniform", lapse_tau=None, lapse_shape=None):
    expected = param_names(stims, model, saturating="c50" in theta)
    if set(theta) != set(expected):
        raise ValueError(f"missing parameters: {sorted(set(expected) - set(theta))}, unknown parameters: {sorted(set(theta) - set(expected))}")
    diffusion = DiffusionParams(theta["gain"], theta["gamma"], theta["sigma"], theta["t0"], theta.get("c50"))
    gates = {s: GateParams(theta[f"scale_{s}"], theta[f"shape_{s}"], theta["ceiling"]) for s in stims}   # the ceiling is shared
    if model is GatedDiffusion:
        return model(gates, diffusion, task)
    return model(gates, diffusion, LapseParams(theta["lapse_prob"], lapse_dist, lapse_tau, lapse_shape), task)


# the parameters held fixed, checked against the parameter names
def _fixed(fixed, names):
    fixed = dict(fixed or {})
    if unknown := set(fixed) - set(names):
        raise ValueError(f"unknown fixed parameters: {sorted(unknown)}, expected names from {list(names)}")
    return fixed


# --- trial-wise log-likelihood in PyTorch ---
#
# The same maths as `log_likelihood` in model.py, written with tensors so that many parameter
# vectors (one per MCMC chain) are evaluated in a single call.
#
# Every hit is scored on its reaction time, catch trials included. A catch-trial hit is a pure lapse, so
# its term is log(lapse prob) + log(lapse timing density). The timing has no fitted parameters, so catch
# reaction times add a constant and cannot move any estimate.

EPS = torch.finfo(torch.float64).tiny   # floor for probabilities before taking the log
MIN_T = 1e-9    # stands in for times at or before the start of a process (density and cdf are 0 there)


# inverse-Gaussian first-passage pdf and cdf at time t, for mean mu and shape lam
def _fpt(t, mu, lam):
    t = t.clamp(min=MIN_T)
    pdf = torch.sqrt(lam / (2 * math.pi * t ** 3)) * torch.exp(-lam * (t - mu) ** 2 / (2 * mu ** 2 * t))
    a = torch.sqrt(lam / t)
    # second term is exp(2 lam / mu) * ndtr(...), kept in log space so it cannot overflow
    cdf = ndtr(a * (t / mu - 1)) + torch.exp(2 * lam / mu + log_ndtr(-a * (t / mu + 1)))
    return pdf, cdf


# lapse density f_L and survival S_L (no lapse yet) at reaction time rt, given p(lapse in the window) = p
def _lapse(rt, p, floor, window, lapse_dist, lapse_tau, lapse_shape):
    if lapse_dist == "uniform":                     # flat density p / window over the window
        return (p / window).expand(-1, len(rt)), 1 - p * (rt - floor) / window
    if lapse_dist == "invgauss":                    # timed from stimulus onset, restricted to the window
        mu, lam = torch.tensor(lapse_tau, dtype=rt.dtype), torch.tensor(lapse_shape, dtype=rt.dtype)
        start, end = _fpt(torch.tensor([floor, floor + window], dtype=rt.dtype), mu, lam)[1]    # cdf at the window edges
        timing, done = _fpt(rt, mu, lam)
        return p * timing / (end - start), 1 - p * (done - start) / (end - start)
    raise ValueError(f"unknown lapse dist: {lapse_dist}")


# theta: (n_chains, n_params), float64, with one column per entry of `names` (from `param_names`).
#        Without a c50 column the drift is a power law
# stim_index, contrast, hit, rt: one entry per trial, with stim_index the position of the trial's
#        stimulus in the stims list given to `param_names`
# returns the log-likelihood summed over trials, shape (n_chains,)
def log_likelihood(theta, names, stim_index, contrast, hit, rt, model=GatedDiffusion, task=None,
                   lapse_dist="uniform", lapse_tau=None, lapse_shape=None):
    task = task or TaskParams()
    floor, window = task.floor, task.deadline - task.floor

    theta = theta[:, :, None]                       # (n_chains, n_params, 1): broadcasts over trials
    column = {name: theta[:, i] for i, name in enumerate(names)}
    gain, gamma, sigma, ceiling, t0 = (column[name] for name in ("gain", "gamma", "sigma", "ceiling", "t0"))
    # gate parameters of each trial's stimulus, (n_chains, n_trials). The stimuli are in the order of `names`
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

    if model is GatedDiffusion:
        pdf, miss = q * f_E, q * (1 - F_E_end) + (1 - q)
    else:
        p = column["lapse_prob"]
        f_L, S_L = _lapse(rt, p, floor, window, lapse_dist, lapse_tau, lapse_shape)
        if model is GatedDiffusionMixture:
            pdf = q * f_E + (1 - q) * f_L
            miss = q * (1 - F_E_end) + (1 - q) * (1 - p)
        elif model is GatedDiffusionRace:
            pdf = q * (f_E * S_L + f_L * (1 - F_E)) + (1 - q) * f_L
            miss = (q * (1 - F_E_end) + (1 - q)) * (1 - p)
    pdf, miss = pdf / not_early, miss / not_early

    log_hit = torch.log(pdf.clamp(min=EPS))
    log_miss = torch.log(miss.clamp(min=EPS))
    return torch.where(hit, log_hit, log_miss).sum(dim=1)


# --- fits ---

# Gaussian approximation to the posterior around its mode (Laplace), built in sbi's
# unconstrained parameter space so that samples always respect the prior bounds
def _laplace(potential, prior, num_samples, num_starts=500):
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


# trial-level Bayesian fit of behavior: samples from the posterior over the model parameters
# stim, contrast, hit, rt: one entry per trial (rt is ignored on misses)
# method: "nuts" samples the posterior with sbi's NUTS sampler (exact, minutes);
#         "laplace" approximates it with a Gaussian at the mode (seconds, for prototyping)
# lapse_dist, lapse_tau, lapse_shape: as in `build_model`. saturating: as in `param_names`
# fixed: optional dict parameter name -> value of parameters held fixed and not sampled, e.g. lapse_prob
#        at the catch-trial hit rate, or t0 at the floor. The returned samples carry them as constant columns
# The prior is uniform over `param_bounds`.
# returns posterior samples as a DataFrame: one row per draw, one column per parameter of `param_names`
def fit_trials(stim, contrast, hit, rt, model=GatedDiffusion, task=None, method="nuts",
               lapse_dist="uniform", lapse_tau=None, lapse_shape=None, saturating=True, fixed=None,
               num_samples=2000, num_chains=4, warmup_steps=500):
    stims = list(np.unique(stim))
    names = param_names(stims, model, saturating)
    fixed = _fixed(fixed, names)
    stim_index = torch.as_tensor(np.searchsorted(stims, stim))
    contrast, hit, rt = (torch.tensor(np.asarray(x)) for x in (contrast, hit, rt))     # copies: the inputs may be read-only
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
        return prior.log_prob(theta) + log_likelihood(complete(theta), names, stim_index, contrast, hit, rt, model, task,
                                                      lapse_dist, lapse_tau, lapse_shape)

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


# MAP estimate: the prior is uniform, so it is the parameters of highest likelihood.
# Found by climbing from the posterior mean over the fitted parameters
# posterior: samples from `fit_trials`. fitted: names of the fitted parameters
# build: parameters -> model, with the task window and lapse shape used in the fit
# returns the parameters (name -> value) and their log-likelihood
def map_estimate(posterior, fitted, build, stim, contrast, hit, rt):
    start = posterior.mean().row(0, named=True)

    def neg_log_lik(x):
        value = -build(start | dict(zip(fitted, x))).log_likelihood(stim, contrast, hit, rt)
        return value if np.isfinite(value) else np.inf

    result = minimize(neg_log_lik, [start[name] for name in fitted], method="Nelder-Mead", options=dict(xatol=1e-7, fatol=1e-7, maxiter=5000))
    return start | dict(zip(fitted, (float(value) for value in result.x))), -result.fun
