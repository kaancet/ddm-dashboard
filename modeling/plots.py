import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from sbi.analysis import pairplot
from scipy.optimize import curve_fit


# three-parameter Weibull for the display curves. No floor term: percent correct is
# baseline-normalised, so it is 0 at zero contrast (as in the manuscript's eq. 5).
def weibull(c, alpha, beta, lapse):
    return (1 - lapse) * (1 - np.exp(-(c / alpha) ** beta))


def weibull_curve(c_pct, percent_correct):
    params, _ = curve_fit(weibull, c_pct, percent_correct / 100, p0=[8, 2, 0.08], bounds=([0.1, 0.1, 0], [400, 20, 0.5]))
    xx = np.logspace(0, 2, 300)
    return xx, weibull(xx, *params) * 100


# baseline-normalised percent correct and median reaction time of hits per animal,
# then mean and SD over animals, per stimulus and contrast
# trials: animalid, stim_type, contrast, hit, rt (s)
def summarise(trials):
    baseline = trials.filter(pl.col("contrast") == 0).group_by("animalid").agg(pl.col("hit").mean().alias("baseline"))
    per_animal = (trials.filter(pl.col("contrast") > 0)
        .group_by(["animalid", "contrast", "stim_type"])
        .agg(pl.col("hit").mean().alias("hit_rate"), (pl.col("rt").filter(pl.col("hit")).median() * 1000).alias("median_rt"))
        .join(baseline, on="animalid")
        .with_columns(((pl.col("hit_rate") - pl.col("baseline")) / (1 - pl.col("baseline")) * 100).alias("percent_correct")))
    return (per_animal.group_by(["contrast", "stim_type"])
        .agg(pl.col("percent_correct").mean().alias("pc_mean"), pl.col("percent_correct").std().alias("pc_sd"),
             pl.col("median_rt").mean().alias("rt_mean"), pl.col("median_rt").std().alias("rt_sd"),
             pl.len().alias("n"))
        .sort(["stim_type", "contrast"]))


# marginals on the diagonal, joint contours above it, MAP estimate in red
# posterior: samples, one column per parameter. theta_map: dict parameter name -> value
# order: names of the parameters to draw
def posterior_pairplot(posterior, theta_map, order, figsize=(6, 6)):
    labels = [name.replace("_", "\n") for name in order]
    fig, axes = pairplot(posterior.select(order).to_numpy(), upper="contour", diag="kde", labels=labels, figsize=figsize,
                         diag_kwargs=dict(mpl_kwargs=dict(color="k")), upper_kwargs=dict(mpl_kwargs=dict(colors="k")),
                         points=np.array([[theta_map[name] for name in order]]), fig_kwargs=dict(points_colors=["#D62728"], points_offdiag=dict(marker="+", markersize=10, mew=1.5)))
    fig.legend(handles=[plt.Line2D([], [], color="#D62728", label="MAP")], loc="lower left", bbox_to_anchor=(0.12, 0.15), frameon=False)
    return fig


# top row: summary points (mean ± SD over animals) with a Weibull display curve, for the data ("Empirical")
#   and for simulated datasets ("Model"): one dataset per posterior draw, with the same trials as the data,
#   summarised like the data and averaged
# bottom rows: reaction times of hits per stimulus and contrast (catch trials are the 0% column).
#   Bars are the data, lines are model densities. Each is normalised to unit area
# trials: animalid, stim_type, contrast, hit, rt (s). build: parameters (name -> value) -> model
# draws: posterior draws to simulate from, one row per simulated dataset
# lines: dict label -> (model, line style), the densities drawn in the bottom rows
def predictive_figure(trials, build, draws, lines, palette):
    stim, contrast = trials["stim_type"].to_numpy().astype(str), trials["contrast"].to_numpy()
    summaries = []
    for theta in draws.iter_rows(named=True):
        sim_hit, sim_rt = build(theta).simulate(stim, contrast)
        summaries.append(summarise(trials.with_columns(pl.Series("hit", sim_hit[0]), pl.Series("rt", sim_rt[0]))))
    sim_summary = pl.concat(summaries).group_by(["contrast", "stim_type"]).mean().sort(["stim_type", "contrast"])

    contrasts = np.sort(trials["contrast"].unique().to_numpy())

    # a 28-column grid holds 4 panels in the top row and one panel per contrast (7) in the two rows below
    fig = plt.figure(figsize=(14, 8.4))
    grid = fig.add_gridspec(3, 28, height_ratios=[1.5, 1, 1], left=0.06, right=0.99, top=0.96, bottom=0.07, hspace=0.5, wspace=0.8)

    # top row
    axes = [fig.add_subplot(grid[0, 7 * i + (i > 0) : 7 * i + 6]) for i in range(4)]
    for (ax_pc, ax_rt), (label, summary) in zip([axes[:2], axes[2:]], [("Empirical", summarise(trials)), ("Model", sim_summary)]):
        for stim_type, color in palette.items():
            s = summary.filter(pl.col("stim_type") == stim_type)
            c_pct = s["contrast"].to_numpy() * 100
            ax_pc.errorbar(c_pct, s["pc_mean"], yerr=s["pc_sd"], fmt="o", color=color, ms=5, lw=1.2)
            ax_pc.plot(*weibull_curve(c_pct, s["pc_mean"].to_numpy()), color=color, lw=1.8)
            ax_rt.errorbar(c_pct, s["rt_mean"], yerr=s["rt_sd"], fmt="o-", color=color, ms=5, lw=1.2)
        ax_pc.set_title(label, loc="left", fontsize=10)
        ax_pc.text(10, 2, f"n={summary['n'][0]:.0f}", ha="center", fontsize=9)
        ax_pc.set(xscale="log", xlim=(1, 130), ylim=(0, 100), ylabel="Percent correct (%)", yticks=[0, 25, 50, 75, 100])
        ax_rt.set(xscale="log", yscale="log", xlim=(1, 130), ylim=(100, 1000), ylabel="Median reaction time (ms)")
        ax_rt.set_yticks([100, 200, 400, 600, 800, 1000])
        ax_rt.set_yticklabels([100, 200, 400, 600, 800, 1000])
        ax_rt.yaxis.set_minor_locator(plt.NullLocator())
        for ax in (ax_pc, ax_rt):
            ax.set_xlabel("Contrast (%)")
            ax.spines[["top", "right"]].set_visible(False)

    # bottom rows, over the observation window of the models
    task = next(iter(lines.values()))[0].task
    bins = np.arange(task.floor, task.deadline + 1e-9, 0.035)
    rt_grid = np.linspace(task.floor + 1e-4, task.deadline, 300)

    axes = np.empty((2, len(contrasts)), dtype=object)
    for i in range(2):
        for j in range(len(contrasts)):
            axes[i, j] = fig.add_subplot(grid[i + 1, 4 * j : 4 * j + 4], sharex=axes[0, 0], sharey=axes[0, 0])
            axes[i, j].tick_params(labelbottom=i == 1, labelleft=j == 0)
    for row, (stim_type, color) in zip(axes, palette.items()):
        for ax, c in zip(row, contrasts):
            empirical = trials.filter((pl.col("stim_type") == stim_type) & (pl.col("contrast") == c) & pl.col("hit"))["rt"].to_numpy()
            ax.hist(empirical, bins=bins, density=True, color=color, alpha=0.45, label="data")
            cc = np.array([c])
            for label, (m, style) in lines.items():
                ax.plot(rt_grid, m.rt_pdf(stim_type, cc, rt_grid) / m.hit_prob(stim_type, cc), color="k", lw=1.2, ls=style, label=label)
            ax.set_title(f"{c * 100:.3g}%  n={len(empirical)}", fontsize=8)
            ax.spines[["top", "right"]].set_visible(False)
        row[0].set_ylabel(f"{stim_type}\ndensity", fontsize=9)
    for ax in axes[1]:
        ax.set_xlabel("RT (s)")
    axes[0, -1].legend(frameon=False, fontsize=7)
    return fig


# 0% has no place on a log axis: it is drawn at X0 (in %), left of the curves, past a break in the axis
X0, X_BREAK = 0.003, 0.006


def _zero_contrast_axis(ax):
    ax.set_xlim(0.002, 130)
    ax.set_xticks([X0, 0.01, 0.1, 1, 10, 100], ["0", "0.01", "0.1", "1", "10", "100"])
    ax.set_xticks([k * 10.0 ** e for e in (-2, -1, 0, 1) for k in range(2, 10)], minor=True)
    # two short slashes across the x axis mark the break
    for dx in (0.93, 1.07):
        ax.plot([X_BREAK * dx / 1.08, X_BREAK * dx * 1.08], [-0.025, 0.025], color="k", lw=0.8,
                transform=ax.get_xaxis_transform(), clip_on=False)


# left: gate probability, one curve per stimulus, dashed line at its scale, dotted line at the ceiling
# right: drift, shared by both stimuli, dashed line at c50, dotted line at the gain
# model: a model with a saturating drift. contrasts: the tested contrasts, marked with dots
def gate_drift_figure(model, contrasts, palette):
    c_fine = np.logspace(-4, 0, 400)        # down to 0.01%, where the gates are close to 0
    tested = contrasts[contrasts > 0]
    zero = np.array([0.0])
    gain, c50 = model.diffusion.gain, model.diffusion.c50
    ceiling = next(iter(model.gates.values())).ceiling

    fig, (ax_gate, ax_drift) = plt.subplots(1, 2, figsize=(9, 3.6), constrained_layout=True)

    for stim_type, color in palette.items():
        ax_gate.plot(c_fine * 100, model.gate_prob(stim_type, c_fine), color=color, lw=1.8, label=stim_type)
        ax_gate.plot(tested * 100, model.gate_prob(stim_type, tested), "o", color=color, ms=4)
        ax_gate.plot(X0, model.gate_prob(stim_type, zero), "o", color=color, ms=4, clip_on=False)
        ax_gate.axvline(model.gates[stim_type].scale * 100, color=color, ls="--", lw=0.8)
    ax_gate.axhline(ceiling, color="0.5", ls=":", lw=0.8)
    ax_gate.text(c_fine[0] * 110, ceiling + 0.02, f"ceiling {ceiling:.2f}", fontsize=8, color="0.4")
    ax_gate.set(xscale="log", ylim=(0, 1), xlabel="Contrast (%)", ylabel="Gate probability q(c)", title="Gate")
    ax_gate.legend(frameon=False, fontsize=8, loc="lower right")

    ax_drift.plot(c_fine * 100, model.drift(c_fine), color="k", lw=1.8)
    ax_drift.plot(tested * 100, model.drift(tested), "o", color="k", ms=4)
    ax_drift.plot(X0, model.drift(zero), "o", color="k", ms=4, clip_on=False)
    ax_drift.axhline(gain, color="0.5", ls=":", lw=0.8)
    ax_drift.text(c_fine[0] * 110, gain * 1.02, f"gain {gain:.2f}", fontsize=8, color="0.4", va="bottom")
    ax_drift.axvline(c50 * 100, color="k", ls="--", lw=0.8)
    ax_drift.text(c50 * 108, 0.3, f"c50 {c50 * 100:.1f}%", fontsize=8)
    ax_drift.set(xscale="log", ylim=(0, gain * 1.12), xlabel="Contrast (%)", ylabel="Drift v(c)", title="Drift (Naka-Rushton)")

    for ax in (ax_gate, ax_drift):
        _zero_contrast_axis(ax)
        ax.spines[["top", "right"]].set_visible(False)
    return fig
