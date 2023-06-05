from __future__ import annotations
import re
import os
from pathlib import Path
import argparse
from typing import Any, Iterable, TYPE_CHECKING, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon
import cv2
from tqdm import tqdm

if TYPE_CHECKING:
    from numpy import ndarray


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
    return parser


def generate_viz(input_file: str, output_dir: str, skip_n: int):
    input_file_path = Path(input_file)
    with open(input_file_path, "r") as log_file:
        log_content = log_file.readlines()

    output_dir_path = Path(output_dir)
    graph_dir = output_dir_path / "graphs"
    prepare_output_folders(output_dir_path, graph_dir)

    files = graph_dir.glob("*")
    video_file = output_dir_path / "project.avi"
    remove_existing_content(files, video_file)

    log_obstacles = log_content[0]
    log_content_filtered = log_content[1 :: skip_n + 1]
    obstacles = parse_obstacles(log_obstacles)
    for i, content in enumerate(tqdm(log_content_filtered)):
        prepare_plot()
        plot_obstacles(obstacles)
        _, agent_positions = parse_simulation_epoch(content)
        plot_agents(agent_positions)
        plt.savefig(graph_dir / f"{i}.png")

    sorted_files = sorted(
        graph_dir.glob("*.png"),
        key=lambda x: float(re.findall(r"(\d+)", str(x))[0]),
    )

    img_array = []
    for filename in tqdm(sorted_files):
        img = cv2.imread(str(filename))
        img_array.append(img)

    create_and_save_video_from_images(img_array, video_file)


def prepare_output_folders(output_dir_path: Path, graph_dir: Path):
    if not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path)

    if not os.path.exists(graph_dir):
        os.mkdir(graph_dir)


def remove_existing_content(files: Iterable[Path], video_file: Path):
    for f in files:
        f.unlink()
    if video_file.exists():
        video_file.unlink()


def parse_obstacles(content: str):
    content_without_newline = content[:-1]
    obstacles = content_without_newline.split("|")
    obstacles_list = []
    for obstacle in obstacles:
        obstacle_without_brackets = obstacle[1:-1]
        split_obstacle_vertex = obstacle_without_brackets.split()
        points = parse_points(split_obstacle_vertex)
        obstacles_list.append(points)
    return obstacles_list


def parse_points(content: str) -> Iterable[Tuple[int, int]]:
    points = []
    for point_2d in content:
        point_2d_no_parenthesis = point_2d[1:-1]
        point_2d_separated_by_comma = point_2d_no_parenthesis.split(",")
        clean_point_2d = (
            float(point_2d_separated_by_comma[0]),
            float(point_2d_separated_by_comma[1]),
        )
        points.append(clean_point_2d)
    return points


def parse_simulation_epoch(content: str):
    content_splitted = content.split()
    global_time = content_splitted[0]
    position_content = content_splitted[1:]
    agent_positions = parse_points(position_content)
    return global_time, agent_positions


def prepare_plot():
    plt.clf()
    # plt.axis(option="on")
    # plt.axis()
    plt.xlim(0, 25)
    plt.ylim(-10, 20)
    plt.axis("equal")


def plot_obstacles(obstacles: Iterable[Iterable[Tuple[int, int]]]):
    ax = plt.gca()
    for obstacle in obstacles:
        ax.add_patch(Polygon(obstacle))


def plot_agents(agent_positions: Iterable[Tuple[int, int]]):
    ax = plt.gca()
    for position in agent_positions:
        ax.add_patch(Circle((position[0], position[1]), 2, color="b", fill=False))


def create_and_save_video_from_images(images: Iterable[ndarray], video_file: Path):
    if not images:
        return
    height, width, _ = images[0].shape
    size = (width, height)
    out = cv2.VideoWriter(str(video_file), cv2.VideoWriter_fourcc(*"DIVX"), 15, size)

    for item in images:
        out.write(item)
    out.release()


if __name__ == "__main__":
    main()
