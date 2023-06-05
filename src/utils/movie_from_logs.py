import glob
import re
import os
import argparse

import matplotlib.pyplot as plt
import numpy as np
import cv2
from tqdm import tqdm


def main():
    parser = get_parser()
    arguments = parser.parse_args()

    generate_viz(arguments.i, arguments.o, arguments.s)


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-i", type=str, default=None, help="path to orca log config file", required=True
    )
    parser.add_argument(
        "-o",
        type=str,
        default=None,
        help="folder to put generated viz graphs and video",
        required=True,
    )
    parser.add_argument(
        "-s", type=int, default=0, help="frames skipped", required=False
    )


def generate_viz(input_file, output_dir, skip_n):
    with open(input_file, "r") as orca:
        run = [line.split("\n") for line in orca.read().split("\n\n")][:-1]

    orca_lines = [
        [list(map(lambda x: float(x), line.split(", "))) for line in r] for r in run
    ]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    graph_dir = os.path.join(output_dir, "graphs")
    if not os.path.exists(graph_dir):
        os.mkdir(graph_dir)

    # draw(graph_dir, 0, orca_lines[-1], False)

    files = glob.glob(os.path.join(graph_dir, "*"))
    video_file = os.path.join(output_dir, "project.avi")
    for f in files:
        os.remove(f)
    if os.path.exists(video_file):
        os.remove(video_file)

    enum_orca = orca_lines[:: skip_n + 1]
    for idx, line in enumerate(tqdm(enum_orca)):
        draw(graph_dir, idx, line, True)
    img_array = []
    if os.path.exists("project.avi"):
        os.remove("project.avi")

    size = None
    sorted_files = sorted(
        glob.glob(os.path.join(graph_dir, "*.png")),
        key=lambda x: float(re.findall(r"(\d+)", x)[0]),
    )

    for filename in tqdm(sorted_files):
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)

    out = cv2.VideoWriter(video_file, cv2.VideoWriter_fourcc(*"DIVX"), 15, size)

    for item in img_array:
        out.write(item)
    out.release()


def draw(graph_dir, num, lines, save=False):
    plt.clf()
    # plt.axis(option="on")
    plt.axis()
    min_lim = -0.5
    max_lim = 0.5
    plt.xlim(min_lim, max_lim)
    plt.ylim(min_lim, max_lim)
    plt.scatter(0, 0, color="b")
    num_lines = len(lines) - 4
    for i, line in enumerate(lines):
        if i == 0:
            continue
        if i < num_lines:
            angle = np.arctan2(line[3], line[2])
            point1 = np.array([line[0], line[1]]) + 4.0 * np.array(
                [np.cos(angle), np.sin(angle)]
            )
            point2 = np.array([line[0], line[1]]) - 4.0 * np.array(
                [np.cos(angle), np.sin(angle)]
            )
            plt.plot([point1[0], point2[0]], [point1[1], point2[1]], color="y", lw=0.8)
            diff = point1 - line[:2]
            plt.arrow(
                point2[0],
                point2[1],
                diff[0],
                diff[1],
                head_width=0.04,
                head_length=0.1,
                fc="k",
                ec="k",
            )
        elif i == len(lines) - 4:
            plt.scatter(line[0], line[1], c="g", alpha=0.6)
            plt.plot([0, line[0]], [0, line[1]], "r-", label="mapping velocity")
        elif i == len(lines) - 3:
            plt.scatter(line[0], line[1], c="g", alpha=0.6)
            plt.plot([0, line[0]], [0, line[1]], "g--+", label="preferred velocity")
        elif i == len(lines) - 2:
            plt.plot([0, line[0]], [0, line[1]], "b--+", label="orca velocity")
        else:
            plt.plot(
                [0, np.cos(line[0])],
                [0, np.sin(line[0])],
                "bv",
                label="orca heading",
            )
            # pass

    area_points = intersect_lines(lines[1:num_lines], min_lim, max_lim)
    plt.scatter(
        [p[0] for p in area_points],
        [p[1] for p in area_points],
        c="g",
        alpha=0.4,
        marker="x",
        label="available region",
    )
    plt.legend()
    if save:
        plt.savefig(os.path.join(graph_dir, f"{num}.png"))
    else:
        plt.show()


def intersect_lines(lines, min_x, max_x):
    x = np.linspace(min_x, max_x, 50)
    y = np.linspace(min_x, max_x, 50)
    points = np.dstack(np.meshgrid(x, y)).reshape(-1, 2)
    for line in lines:
        line_point, line_direction = np.array(line[:2]), np.array(line[2:])
        # points on the left of the direction of the line
        points = np.array(
            [
                point
                for point in points
                if np.cross(point - line_point, line_direction) <= 0
            ]
        )
    return points


if __name__ == "__main__":
    main()
