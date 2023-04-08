from __future__ import annotations

import csv
import os
from collections.abc import Generator

import matplotlib.pyplot as plt  # type: ignore[import]
import numpy as np


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


def bar_plot(file: str) -> None:
    assert file.endswith(".csv")
    # Define the data
    models = ['model_a', 'model_b_rebasin', 'model_b_original']

    y_values = [0.0, 0.0, 0.0]
    filename = full_filename(file)
    with open(filename) as f_:
        reader = csv.reader(f_)
        for i, row in enumerate(reader):
            if i == 0:
                continue
            if i == 1:
                y_values[0] = float(row[1])
                y_values[2] = float(row[3])
            else:
                y_values[1] = float(row[2])

    # Create a figure
    fig = plt.figure()
    ax = plt.subplot(111)

    # Create a bar chart
    colors = ['red', 'green', 'blue']
    bars = ax.bar(models, y_values, width=0.5, align='center', color=colors)

    # Hide the x-ticks
    plt.xticks([])

    # Add labels to the bars
    for i, (m, v) in enumerate(zip(models, y_values)):  # noqa
        plt.text(i, v, str(round(v, 2)), ha='center', va='bottom')

    # Add a y-axis label
    plt.ylabel('Validation Loss')

    # Shrink box to allow legend to be placed outside of it
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width, box.height * 0.95])

    # Create a legend
    plt.legend(
        bars, models, loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=len(models)
    )

    # Adjust the x-axis limits to show all the bars
    plt.xlim(-0.5, len(models)-0.5)

    # Save the plot
    filename = to_named_img(file, "bar")
    fig.savefig(full_filename(filename), dpi=300)


def line_plot(file: str) -> None:
    assert file.endswith(".csv")
    # Define the data
    models = ['a-b(orig)', 'a-b(rebasin)', 'b(orig)-b(rebasin)']

    y_values: list[list[float]] = []
    filename = full_filename(file)
    with open(filename) as f_:
        reader = csv.reader(f_)
        for i, row in enumerate(reader):
            if i == 0:
                continue
            y_values.append([float(row[1]), float(row[2]), float(row[3])])

    max_loss = 0.0
    for row in y_values:  # type: ignore[assignment]
        max_loss = max(max_loss, np.max(row))

    xs = np.arange(0, len(y_values), 1)

    # Create a figure
    fig = plt.figure()
    ax = plt.subplot(111)
    plt.ylim(0.0, np.ceil(max_loss))
    plt.grid(True)

    # Create the plot
    lines = ax.plot(xs, y_values)

    # Add an x- and y-axis label
    plt.ylabel('Validation Loss')
    plt.xlabel('Interpolation: % model_b')

    # Shrink box to allow legend to be placed outside of it
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width, box.height * 0.95])

    # Create a legend
    plt.legend(
        lines, models, loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=len(models)
    )

    # Save the plot
    filename = to_named_img(file, "line")
    #fig.savefig(full_filename(filename), dpi=300)
    plt.show()



def file_generator() -> Generator[str, None, None]:
    for file in os.listdir(full_filename("")):  # lists files in results
        if file.endswith(".csv"):
            yield file


if __name__ == "__main__":
    for f in file_generator():
        bar_plot(f)
        line_plot(f)
