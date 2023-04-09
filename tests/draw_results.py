from __future__ import annotations

import csv
import os
from collections.abc import Generator
from typing import NamedTuple

import matplotlib.pyplot as plt  # type: ignore[import]
import numpy as np

from tests.fixtures.mandw import MODELS_AND_WEIGHTS


def full_filename(file: str) -> str:
    filename = os.path.dirname(__file__)
    filename = os.path.join(filename, "results")
    filename = os.path.join(filename, file)
    return filename


def to_named_img(file: str, name: str) -> str:
    filename = file.split("/")[-1]  # bla.csv
    filename = os.path.join("images", filename)
    filename = full_filename(filename)
    return f"{filename[:-4]}_{name}.png"


def file_generator() -> Generator[str, None, None]:
    for file in os.listdir(full_filename("")):  # lists files in results
        if file.endswith(".csv"):
            yield file


class SweepInfo(NamedTuple):
    """Stores information about an interpolation sweep."""
    model: str
    w_start: str
    w_end: str
    losses: list[float]


def get_info(file: str) -> tuple[SweepInfo, SweepInfo, SweepInfo]:
    assert file.endswith(".csv")
    file = file.split("/")[-1]  # only get model name

    model_name = file[:-4]  # remove .csv
    minfo = [m for m in MODELS_AND_WEIGHTS if m.constructor.__name__ == model_name][0]
    wa = str(minfo.weights_a)
    wb_orig = str(minfo.weights_b)
    wb_rebasin = wb_orig + "_REBASIN"

    sweep_a_b_orig: list[float] = []
    sweep_a_b_rebasin: list[float] = []
    sweep_b_orig_b_rebasin: list[float] = []

    with open(full_filename(file)) as f_:
            reader = csv.reader(f_)
            for i, row in enumerate(reader):
                if i == 0:
                    continue
                else:
                    sweep_a_b_orig.append(float(row[1]))
                    sweep_a_b_rebasin.append(float(row[2]))
                    sweep_b_orig_b_rebasin.append(float(row[3]))

    return (
        SweepInfo(model_name, wa, wb_orig, sweep_a_b_orig),
        SweepInfo(model_name, wa, wb_rebasin, sweep_a_b_rebasin),
        SweepInfo(model_name, wb_orig, wb_rebasin, sweep_b_orig_b_rebasin),
    )


def bar_plot(file: str) -> None:
    sweep_ab_orig, sweep_ab_rebasin, sweep_b_orig_b_rebasin = get_info(file)

    loss_a = sweep_ab_orig.losses[0]
    loss_b_orig = sweep_ab_orig.losses[-1]
    loss_b_rebasin = sweep_ab_rebasin.losses[-1]
    y_values = [loss_a, loss_b_rebasin, loss_b_orig]

    model_a_name = str(sweep_ab_orig.w_start)
    model_b_orig_name = str(sweep_ab_orig.w_end)
    model_b_rebasin_name = str(sweep_ab_rebasin.w_end)
    model_names = [model_a_name, model_b_rebasin_name, model_b_orig_name]

    # Create a figure
    fig = plt.figure()
    ax = plt.subplot(111)

    # Create a bar chart
    colors = ['red', 'green', 'blue']
    bars = ax.bar(model_names, y_values, width=0.5, align='center', color=colors)

    # Hide the x-ticks
    plt.xticks([])

    # Add labels to the bars
    for i, (m, v) in enumerate(zip(model_names, y_values)):  # noqa
        plt.text(i, v, str(round(v, 2)), ha='center', va='bottom')

    # Add a y-axis label
    plt.ylabel('Validation Loss')

    # Shrink box to allow legend to be placed outside of it
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width, box.height * 0.9])

    # Create a legend
    plt.legend(
        bars, model_names, loc='lower center', bbox_to_anchor=(0.5, 1.0)
    )

    # Adjust the x-axis limits to show all the bars
    plt.xlim(-0.5, len(model_names) - 0.5)

    # Save the plot
    filename = to_named_img(file, "bar")
    fig.savefig(full_filename(filename), dpi=300)
    # plt.show()


def line_plot(file: str) -> None:
    sweep_ab_orig, sweep_ab_rebasin, sweep_b_orig_b_rebasin = get_info(file)
    sweeps = sweep_ab_orig, sweep_ab_rebasin, sweep_b_orig_b_rebasin
    sweep_names = ['a-b(orig)', 'a-b(rebasin)', 'b(orig)-b(rebasin)']

    model_a_name = str(sweep_ab_orig.w_start)
    model_b_orig_name = str(sweep_ab_orig.w_end)
    model_b_rebasin_name = str(sweep_ab_rebasin.w_end)

    # Get axes
    fig, axs = plt.subplots(
        nrows=3, ncols=1, figsize=(6, 8), sharex='all', gridspec_kw={'hspace': 0.7}
    )

    # Set tiles
    fig.suptitle(
        f"Interpolation between models of type {sweep_ab_orig.model}",
        fontsize=14,
        fontweight='bold'
    )
    axs[0].set_title(
        f"w_a: {model_a_name} \nw_b: {model_b_orig_name}", loc="left", fontsize=10
    )
    axs[1].set_title(
        f"w_a: {model_a_name} \nw_b: {model_b_rebasin_name}", loc="left", fontsize=10
    )
    axs[2].set_title(
        f"w_a: {model_b_orig_name} \nw_b: {model_b_rebasin_name}",
        loc="left",
        fontsize=10
    )

    # Define ylims
    maxloss_abo = np.max(sweep_ab_orig.losses)
    maxloss_abr = np.max(sweep_ab_rebasin.losses)
    maxloss_bobr = np.max(sweep_b_orig_b_rebasin.losses)
    maxloss = np.max([maxloss_abo, maxloss_abr, maxloss_bobr]) * 1.1
    ylim_upper = np.ceil(maxloss)

    minloss_abo = np.min(sweep_ab_orig.losses)
    minloss_abr = np.min(sweep_ab_rebasin.losses)
    minloss_bobr = np.min(sweep_b_orig_b_rebasin.losses)
    minloss = np.min([minloss_abo, minloss_abr, minloss_bobr]) * 0.9
    ylim_lower = np.floor(minloss)

    # Set x-labels
    xlabels = np.arange(len(sweep_ab_orig.losses), dtype=np.float64)
    xlabels /= (float(len(sweep_ab_orig.losses)) - 1)
    xlabels *= 100.0  # convert to percentage

    # Plot
    for i, ax in enumerate(axs):
        ax.set_ylim(ylim_lower, ylim_upper)
        ax.plot(xlabels, sweeps[i].losses, label=sweep_names[i], color='gray')
        ax.set_ylabel('Validation Loss')
        ax.grid(True)

        loss0 = sweeps[i].losses[0]
        loss1 = sweeps[i].losses[-1]

        # Mark the first and last point,
        #   as well as the point of minimum loss,
        #   in the plot with a circle
        oa = ax.plot(0, loss0, 'o', color='red', label=f"w_a: {loss0:.2f}")
        ob = ax.plot(xlabels[-1], loss1, 'o', color='blue', label=f"w_b: {loss1:.2f}")

        # Get the index of the minimum loss
        #   and plot & label it if it's not the first or last point
        minloss_idx = np.argmin(sweeps[i].losses)
        minloss = sweeps[i].losses[minloss_idx]
        omin = ax.plot(
            xlabels[minloss_idx],
            sweeps[i].losses[minloss_idx],
            'o',
            color='green',
            label=f"min: {minloss:.2f}"
        )

        # Add a legend
        objs = [oa[0], omin[0], ob[0]]
        labels = [f"w_a: {loss0:.2f}", f"min: {minloss:.2f}", f"w_b: {loss1:.2f}"]
        colors = ['red', 'green', 'blue']

        ax.legend(
            objs,
            labels,
            loc='upper center',
            bbox_to_anchor=(0.5, -0.3 if i==2 else -0.05),
            labelcolor=colors,
            ncol=3
        )

    axs[0].set_xticks([])
    axs[1].set_xticks([])

    # I want 0 and 100 to be included in the xticks, and the rest of the
    #   ticks to be spaced evenly.
    dist_between_ticks = int((len(xlabels) - 1) / len(xlabels) * 4)
    xticks = list(xlabels[3 : -3 : 3])
    xticks = [0, *xticks] + [100]
    axs[2].set_xticks(xticks)

    plt.xlabel('Interpolation: % w_b')

    # Save the plot
    filename = to_named_img(file, "line")
    fig.savefig(full_filename(filename), dpi=300)


if __name__ == "__main__":
    for f in file_generator():
        bar_plot(f)
        line_plot(f)
