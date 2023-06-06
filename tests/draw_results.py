from __future__ import annotations

import csv
import os
from collections.abc import Generator
from typing import Any, NamedTuple

import matplotlib.pyplot as plt  # type: ignore[import]
import numpy as np

from tests.fixtures.mandw import MODELS_AND_WEIGHTS


def full_filename(file: str, lib: str, dataset: str = "") -> str:
    filename = os.path.dirname(__file__)
    filename = os.path.join(filename, "results")
    filename = os.path.join(filename, lib) if lib else filename
    filename = os.path.join(filename, dataset) if dataset else filename
    filename = os.path.join(filename, file) if file else filename
    return filename


def to_named_img(file: str, name: str, lib: str, dataset: str = "") -> str:
    filename = file.split("/")[-1]  # bla.csv
    filename = os.path.join("images", filename)
    filename = full_filename(filename, lib, dataset)
    return f"{filename[:-4]}_{name}.png"


def file_generator(lib: str, dataset: str = "") -> Generator[str, None, None]:
    for file in os.listdir(full_filename("", lib, dataset)):  # list files in results
        if file.endswith(".csv"):
            yield file


class SweepInfo(NamedTuple):
    """Stores information about an interpolation sweep."""
    model: str
    w_start: str
    w_end: str
    losses: list[float]


def get_info(file: str, dataset: str) -> tuple[SweepInfo, SweepInfo, SweepInfo]:
    assert file.endswith(".csv")
    file = file.split("/")[-1]  # only get model name

    model_name = file[:-4]  # remove .csv
    minfo = [m for m in MODELS_AND_WEIGHTS if m.constructor.__name__ == model_name][0]
    wa = str(minfo.weights_a)
    wb = str(minfo.weights_b)
    wb_orig = wb + " (orig)"
    wb_rebasin = wb + " (rebasin)"

    sweep_a_b_orig: list[float] = []
    sweep_a_b_rebasin: list[float] = []
    sweep_b_orig_b_rebasin: list[float] = []

    with open(full_filename(file, "torchvision", dataset)) as f_:
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


def bar_plot(file: str, dataset: str = "cifar10") -> None:
    sweep_ab_orig, sweep_ab_rebasin, sweep_b_orig_b_rebasin = get_info(file, dataset)

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
    filename = to_named_img(file, "bar", "torchvision", dataset)
    fig.savefig(filename, dpi=300)

    # Close the plot
    plt.close(fig)


def line_plot(file: str, dataset: str = "cifar10") -> None:
    sweep_ab_orig, sweep_ab_rebasin, sweep_b_orig_b_rebasin = get_info(file, dataset)
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
        f"Interpolation: {sweep_ab_orig.model}",
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
        minlabel = f"min: {minloss:.2f} (%={xlabels[minloss_idx]:.2f})"
        omin = ax.plot(
            xlabels[minloss_idx],
            sweeps[i].losses[minloss_idx],
            'o',
            color='green',
            label=minlabel
        )

        # Add a legend
        objs = [oa[0], omin[0], ob[0]]
        labels = [f"w_a: {loss0:.2f}", minlabel, f"w_b: {loss1:.2f}"]
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
    xticks = [0, *xticks, 100]
    axs[2].set_xticks(xticks)

    plt.xlabel('Interpolation: % w_b')

    # Save the plot
    filename = to_named_img(file, "line", "torchvision", dataset)
    fig.savefig(filename, dpi=300)

    # Close the plot
    plt.close(fig)


def read_csv(file: str) -> dict[str, list[Any]]:
    assert file.endswith(".csv")
    with open(file) as f:
        reader = csv.reader(f)
        header = next(reader)
        data: dict[str, list[str | float | int]] = {h: [] for h in header}
        for row in reader:
            for h, v in zip(header, row):
                data[h].append(float(v))

    return data


def draw_hlb_gpt() -> None:
    """Draw the plots for the HLB-GPT model."""
    lfile = full_filename("hlb-gpt/losses.csv", "", "")
    afile = full_filename("hlb-gpt/accuracies.csv", "", "")
    pfile = full_filename("hlb-gpt/perplexities.csv", "", "")

    losses = read_csv(lfile)
    accuracies = read_csv(afile)
    perplexities = read_csv(pfile)

    # Create fig & axs
    fig, axs = plt.subplots(
        nrows=3, ncols=1, figsize=(6, 8), sharex='all', gridspec_kw={'hspace': 0.7}
    )

    # Set titles
    fig.suptitle("HLB-GPT", fontsize=14, fontweight='bold')
    axs[0].set_title("Losses", loc="center", fontsize=10)
    axs[1].set_title("Perplexities", loc="center", fontsize=10)
    axs[2].set_title("Accuracies", loc="center", fontsize=10)

    # Draw
    draw_ax(1, axs[0], losses, "min", "Validation Loss")
    draw_ax(2, axs[1], perplexities, "min", "Validation Perplexity")
    draw_ax(3, axs[2], accuracies, "max", "Validation Accuracy")

    # xlabel
    plt.xlabel('Interpolation: % w_b')

    # Save the plot
    filename = to_named_img("hlb-gpt", "line", "hlb-gpt")
    fig.savefig(filename, dpi=300)

    # Close the plot
    plt.close(fig)


def draw_ax(
        index: int, ax: Any, data: dict[str, list[Any]], min_or_max: str, ylabel: str
) -> None:
    """Draw the plot for a single ax."""
    # Define ylims
    ymax = np.max([
        np.max(data['ab_orig']).item(),
        np.max(data['ab_rebasin']).item(),
        np.max(data['b_orig_b_rebasin']).item()
    ])
    ymin = np.min([
        np.min(data['ab_orig']).item(),
        np.min(data['ab_rebasin']).item(),
        np.min(data['b_orig_b_rebasin']).item()
    ])
    ylim_upper = ymax * 1.1
    ylim_lower = ymin * 0.9

    # Set scientific notation for large numbers
    if ymax >= 100.0:
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

    # Set x-labels
    xlabels = np.arange(len(data['ab_orig']), dtype=np.float64)
    xlabels /= (float(len(data['ab_orig'])) - 1)
    xlabels *= 100.0  # convert to percentage

    # Write the min/max values
    best_val_abo, best_idx_abo = (
        (np.max(data['ab_orig']), np.argmax(data['ab_orig']))
        if min_or_max == 'max'
        else (np.min(data['ab_orig']), np.argmin(data['ab_orig']))
    )
    best_val_abr, best_idx_abr = (
        (np.max(data['ab_rebasin']), np.argmax(data['ab_rebasin']))
        if min_or_max == 'max'
        else (np.min(data['ab_rebasin']), np.argmin(data['ab_rebasin']))
    )
    best_val_bobr, best_idx_bobr = (
        (np.max(data['b_orig_b_rebasin']), np.argmax(data['b_orig_b_rebasin']))
        if min_or_max == 'max'
        else (np.min(data['b_orig_b_rebasin']), np.argmin(data['b_orig_b_rebasin']))
    )

    label_abo = (
        f"ab_orig: {min_or_max}={best_val_abo:.2f} (%={int(xlabels[best_idx_abo])})"
    )
    label_abr = (
        f"ab_rebasin: {min_or_max}={best_val_abr:.2f} (%={int(xlabels[best_idx_abr])})"
    )
    label_bobr = (
        f"b_orig_b_rebasin: {min_or_max}={best_val_bobr:.2f}"
        f" (%={int(xlabels[best_idx_bobr])})"
    )

    # Plot
    ax.set_ylim(ylim_lower, ylim_upper)
    ax.plot(xlabels, data['ab_orig'], label=label_abo, color='red')
    ax.plot(xlabels, data['ab_rebasin'], label=label_abr, color='green')
    ax.plot(xlabels, data['b_orig_b_rebasin'], label=label_bobr, color='blue')
    ax.set_ylabel(ylabel)
    ax.grid(True)

    # Set legend
    ax.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, -0.3 if index == 3 else -0.05),
        ncol=3,
        fontsize=6
    )


if __name__ == "__main__":
    draw_hlb_gpt()
