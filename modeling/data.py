from pathlib import Path

import numpy as np
import polars as pl
from scipy.optimize import minimize
from scipy.stats import invgauss

from .model import TaskParams

DATA = Path(__file__).parents[1] / "data" / "260923_training_data.parquet"

# Label SFTF conditions by the TF
STIMS = {"0.04cpd_8.0Hz": "fast", "0.16cpd_0.5Hz": "slow"}

# Session used in manscript (from kaan)
# KC300 is excluded: different response window?
SESSIONS = [
    "230227_KC139_detect__no_cam_KC",
    "230120_KC143_detect__no_cam_KC",
    "230403_KC144_detect__no_cam_KC",
    "230609_KC145_detect__no_cam_KC",
    "230511_KC146_detect__no_cam_KC",
    "240506_KC147_detect__no_cam_KC",
    "240429_KC148_detect__no_cam_KC",
    "240424_KC149_detect__no_cam_KC",
    "220928_KC301_detect__no_cam_KC",
    "240718_KC302_detect__no_cam_KC",
    "240711_KC303_detect__no_cam_KC",
    "240807_KC304_detect__no_cam_KC",
]

# Load / clean trials in to [animal, stim, contrast, hit, rt]
# A hit is a rig hit with floor < rt < deadline. This window on reaction time approximates the
# rig's 150-1000 ms window on response time
# TODO: replace the hard deadline with p(movement crosses in time), once confirmed with data collection
def load_trials(sessions=SESSIONS, task=None, path=DATA):
    task = task or TaskParams()
    trials = (pl.read_parquet(path)
        .filter(pl.col("sessiondir").is_in(sessions) & (pl.col("outcome") != "early") & pl.col("contrast").is_not_null())
        .select("animalid", pl.col("stim_type").replace(STIMS), "contrast",
                (pl.col("outcome") == "hit").alias("hit"),
                (pl.col("reaction_time") / 1000).alias("rt")))

    # removes: (1) early resps, incl. hits with rt at or under the floor, (2) hits with no rt
    # recodes: hits with rt at or after the deadline become misses
    no_rt = pl.col("hit") & pl.col("rt").is_null()
    early = pl.col("hit") & (pl.col("rt") <= task.floor)
    late = pl.col("hit") & (pl.col("rt") >= task.deadline)
    print(f"{trials.height} trials: dropping {trials.filter(no_rt).height} hits without a reaction time and {trials.filter(early).height} under the floor, "
          f"recoding {trials.filter(late).height} late hits as misses")
    trials = (trials.filter(~no_rt & ~early)
        .with_columns((pl.col("hit") & ~late).alias("hit"), pl.when(late).then(None).otherwise(pl.col("rt")).alias("rt")))
    print(f"{trials.height} trials from {trials['animalid'].n_unique()} animals")
    return trials


# Compute constants that are measured from the trials and not fitted
def measure_constants(trials, task=None, verbose=True):
    task = task or TaskParams()
    contrast, hit, rt = trials["contrast"].to_numpy(), trials["hit"].to_numpy(), trials["rt"].to_numpy()
    catch_rt = rt[(contrast == 0) & hit]

    # optimize the params of inverse gaussian lapse distribution
    def catch_nll(params):
        mean, shape = params
        timing = invgauss(mean / shape, scale=shape)
        return -np.sum(timing.logpdf(catch_rt)) + len(catch_rt) * np.log(timing.cdf(task.deadline) - timing.cdf(task.floor))

    # parameters of fit lapse distribution
    lapse_tau, lapse_shape = minimize(catch_nll, [0.4, 0.8], method="Nelder-Mead").x
    lapse_prob = hit[contrast == 0].mean()

    # ceiling of the psychometric curves (computed at high contrasts)
    plateau = hit[contrast >= 0.5].mean()
    ceiling = (plateau - lapse_prob) / (1 - lapse_prob)

    if verbose:
        print(f"lapse timing from {len(catch_rt)} catch-trial false alarms: inverse Gaussian, mean {lapse_tau:.3f} s, shape {lapse_shape:.3f}")
        print(f"lapse prob from {np.sum(contrast == 0)} catch trials: {lapse_prob:.3f}")
        print(f"ceiling from {np.sum(contrast >= 0.5)} trials at 50% and 100% contrast: hit rate {plateau:.3f}, ceiling {ceiling:.3f}")

    return dict(lapse_tau=float(lapse_tau), lapse_shape=float(lapse_shape), lapse_prob=float(lapse_prob), ceiling=float(ceiling))
