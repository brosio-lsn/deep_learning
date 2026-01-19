# src/visualization/sudoku_video.py

import os
from typing import List

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


def draw_board(ax, board: np.ndarray):
    ax.clear()
    ax.set_xticks(np.arange(0.5, 9.5, 1))
    ax.set_yticks(np.arange(0.5, 9.5, 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xlim(0, 9)
    ax.set_ylim(9, 0)
    ax.grid(which="both", linestyle="-", linewidth=0.5)

    for i in range(10):
        lw = 2.5 if i % 3 == 0 else 0.5
        ax.axhline(i, xmin=0, xmax=9, linewidth=lw, color="black")
        ax.axvline(i, ymin=0, ymax=9, linewidth=lw, color="black")

    for r in range(9):
        for c in range(9):
            v = board[r, c]
            if v != 0:
                ax.text(
                    c + 0.5,
                    r + 0.7,
                    str(int(v)),
                    ha="center",
                    va="center",
                    fontsize=16,
                )

    ax.set_axis_off()


def save_sudoku_video(
    trajectory: List[np.ndarray],
    out_path: str = "sudoku_solution.gif",
    fps: int = 2,
):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    fig, ax = plt.subplots(figsize=(4, 4))

    def update(frame_idx):
        board = trajectory[frame_idx]
        draw_board(ax, board)
        ax.set_title(f"Step {frame_idx}/{len(trajectory)-1}")

    anim = FuncAnimation(
        fig,
        update,
        frames=len(trajectory),
        interval=1000 // fps,
        repeat=False,
    )

    writer = PillowWriter(fps=fps)
    anim.save(out_path, writer=writer)
    plt.close(fig)
