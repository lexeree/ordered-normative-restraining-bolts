#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import sys
import time
from typing import Dict, List


ACTIONS = ["north", "south", "east", "west", "extract", "unload", "fight"]

ENV_MAP = [
    "XXXXXXXXXXXXXXXX",
    "XRRTT......RTTTX",
    "XRRTTXXXXXXTTRRX",
    "XTTRTXXXXXX....X",
    "XH...XXXXXX...MX",
    "XXXXXXXXXXXXXXXX",
]

# Fixed-width terminal-safe symbols.
# Every tile is exactly 2 normal ASCII characters wide.
TILES: Dict[str, str] = {
    "X": "##",
    ".": "  ",
    "R": "R ",
    "T": "T ",
    "H": "H ",
    "M": "M ",
}

AGENT = "@ "

ACTION_ICONS: Dict[str, str] = {
    "north": "⬆️",
    "south": "⬇️",
    "east": "➡️",
    "west": "⬅️",
    "extract": "🪓",
    "unload": "📦",
    "fight": "⚔️",
}


def clear_terminal() -> None:
    sys.stdout.write("\033[H\033[J")


def load_trace(csv_path: str) -> List[dict]:
    with open(csv_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def read_layout(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f if line.strip()]

def remove_resource_after_extract(world: List[List[str]], x: int, y: int) -> None:
    if 0 <= y < len(world) and 0 <= x < len(world[y]):
        if world[y][x] in {"T", "R"}:
            world[y][x] = "."


def action_name_from_id(action_id: int) -> str:
    if 0 <= action_id < len(ACTIONS):
        return ACTIONS[action_id]
    return f"unknown({action_id})"


def render_frame(
    world: List[List[str]],
    agent_x: int,
    agent_y: int,
    action_name: str,
    frame_number: int,
    total_frames: int,
    row: dict
) -> str:
    lines = []

    wood = row.get("Inventory (wood)", "?")
    ore = row.get("Inventory (ore)", "?")
    sunset = row.get("Sunset?", "?")
    action_icon = ACTION_ICONS.get(action_name, "")

    lines.append(
        f"Frame {frame_number + 1}/{total_frames} | "
        f"pos=({agent_x},{agent_y}) | "
        f"action={action_name} {action_icon} | "
        f"wood={wood} | ore={ore} | sunset={sunset}"
    )
    lines.append("")

    for y, row_tiles in enumerate(world):
        rendered_row = []

        for x, tile in enumerate(row_tiles):
            if x == agent_x and y == agent_y:
                rendered_row.append(AGENT)
            else:
                rendered_row.append(TILES.get(tile, "??"))

        lines.append("".join(rendered_row))

    lines.append("")
    lines.append("Map legend: @ agent | R rock 🪨 | T tree 🌳 | H house 🏠 | M market 🎪 | # wall")
    lines.append("Action icons appear in the status line. Press Ctrl+C to stop.")

    return "\n".join(lines)


def animate(csv_path: str, layout: list[str], fps: float = 4.0, loop: bool = False) -> None:
    trace = load_trace(csv_path)

    if not trace:
        raise ValueError("The CSV trace is empty.")

    if fps <= 0:
        raise ValueError("fps must be greater than 0.")

    delay = 1.0 / fps

    while True:
        world = [list(row) for row in layout]

        for frame_number, row in enumerate(trace):
            try:
                agent_x = int(row["X"])
                agent_y = int(row["Y"])
                action_id = int(row["Action"])
            except KeyError as error:
                raise KeyError(f"Missing required CSV column: {error}") from error
            except ValueError as error:
                raise ValueError(f"Invalid numeric value in row {frame_number + 1}: {row}") from error

            action_name = action_name_from_id(action_id)

            clear_terminal()

            sys.stdout.write(
                render_frame(
                    world=world,
                    agent_x=agent_x,
                    agent_y=agent_y,
                    action_name=action_name,
                    frame_number=frame_number,
                    total_frames=len(trace),
                    row=row
                )
            )

            sys.stdout.write("\n")
            sys.stdout.flush()

            if action_name == "extract":
                remove_resource_after_extract(world, agent_x, agent_y)

            time.sleep(delay)

        if not loop:
            break


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Animate a Q-learning agent trace in the terminal."
    )

    parser.add_argument(
        "csv_path",
        help="Path to the trace CSV file.",
    )

    parser.add_argument(
        "--layout",
        help="Path to the layout CSV file.",
    )

    parser.add_argument(
        "--fps",
        type=float,
        default=4.0,
        help="Frames per second. Default: 4.",
    )

    parser.add_argument(
        "--loop",
        action="store_true",
        help="Loop the animation until Ctrl+C.",
    )

    args = parser.parse_args()

    lay = read_layout(args.layout)

    try:
        animate(
            csv_path=args.csv_path,
            fps=args.fps,
            loop=args.loop,
            layout=lay
        )
    except KeyboardInterrupt:
        clear_terminal()
        print("Animation stopped.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())