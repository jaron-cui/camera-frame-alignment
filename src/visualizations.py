import math
import typing

import torch
from matplotlib import pyplot as plt

from utils import tensor_to_image


def display_frame(frame: torch.Tensor, title: str = None):
    if title is not None:
        plt.title(title)
    plt.imshow(tensor_to_image(frame))
    plt.show()
    plt.figure()


def display_most_aligned_frames(
    frames: torch.Tensor,
    similarity_to_start_frames: typing.List[float],
    top_k: int
):
    column_count = 4
    _, sorted_frame_indices = torch.Tensor(similarity_to_start_frames).sort(descending=True)
    most_similar_frames = frames[sorted_frame_indices[:top_k]]

    # format each image into a grid of subplots
    cols, rows = column_count, max(1, math.ceil(top_k / column_count))
    fig, axes = plt.subplots(
        ncols=cols,
        nrows=rows,
        figsize=(cols, rows),
        gridspec_kw={'hspace': 0, 'wspace': 0}
    )
    for i, frame in enumerate(most_similar_frames):
        x, y = i % column_count, i // column_count
        axes[y, x].imshow(tensor_to_image(frame))
        axes[y, x].set_aspect('equal')
        axes[y, x].set_xticklabels([])
        axes[y, x].set_yticklabels([])
        # axes[y, x].set_title(f'Match #{i + 1}')

    fig.suptitle(f'Top {top_k} Matches from Scan')
    subtitle = ', '.join([str(t.item()) for t in sorted_frame_indices[:top_k]])
    fig.text(0.5, 0.93, subtitle, ha='center', fontsize=6, color='gray')
    plt.tight_layout()
    plt.show()


def display_alignment_scores(similarity_to_start_frames: typing.List[float]):
    plt.title(f'Similarity to Start Frames')
    plt.xlabel('Frame')
    plt.ylabel('Similarity')
    plt.scatter(range(len(similarity_to_start_frames)), similarity_to_start_frames, marker='x')
    plt.show()