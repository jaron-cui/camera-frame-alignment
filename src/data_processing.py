import math
import typing
from pathlib import Path

import cv2
import decord
import einops
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision
from torch import nn
from tqdm import tqdm


def load_encoder(weight_path: str, device) -> nn.Module:
    class CustomModel(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            model = torchvision.models.resnet34(weights=torchvision.models.ResNet34_Weights.IMAGENET1K_V1)
            self.layers = nn.Sequential(*list(model.children())[:-2])

        def forward(self, x):
            return self.layers(x).flatten(start_dim=1)

    return CustomModel().to(device)


def load_video_frames(mp4_path: str, cache_path: str = None) -> torch.Tensor:

    def extract_video_frames() -> torch.Tensor:
        # Taken from min-stretch/data-collection/utils/new_gripper_model.py line 54
        # print(f'Loading frames from {mp4_path}.')
        video_reader = decord.VideoReader(
            mp4_path,
            ctx=decord.cpu(0),
            width=256,
            height=256,
            num_threads=-1,
        )
        frames = []
        for i, frame in enumerate(tqdm(video_reader, desc=f'Loading frames from {mp4_path}')):
            if i % 5 != 0:
                continue
            frames.append(torch.Tensor(frame.asnumpy()))

        frames_tensor = torch.stack(frames)
        frames_tensor = frames_tensor / 255.0
        frames_tensor = einops.rearrange(frames_tensor, "t h w c -> t c w h").flip(dims=[2])
        # frames_tensor = NORMALIZER(frames_tensor)
        return frames_tensor

    return cache_operation(extract_video_frames, cache_path=cache_path)


def tensor_to_image(x: torch.Tensor) -> np.ndarray:
    x = (x.detach().cpu() * 255).type(torch.uint8)
    x = einops.rearrange(x, 'c h w -> h w c')
    x = x.numpy()
    return x


def tensors_to_image(x: torch.Tensor) -> np.ndarray:
    x = (x.detach().cpu() * 255).type(torch.uint8)
    x = einops.rearrange(x, 't b c h w -> t b h w c')
    x = x.numpy()
    return x


def encode_frames(encoder: nn.Module, frames: torch.Tensor, cache_path: str = None, batch_size: int = 256) -> torch.Tensor:
    """
    Expects frames of shape (batch x height x width x channels) and returns
    encodings of shape (batch x encoding_length)
    """
    def encode() -> torch.Tensor:
        dataset = torch.utils.data.TensorDataset(frames)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
        outputs = []
        for batch, in tqdm(dataloader, desc=f'Encoding frames in batches of {batch_size}'):
            output = encoder.forward(batch).detach().squeeze(0).unsqueeze(1)
            outputs.append(output)
        return torch.cat(outputs, dim=0)

    return cache_operation(encode, cache_path=cache_path)


def display_frame(frame: torch.Tensor, title: str = None):
    if title is not None:
        plt.title(title)
    plt.imshow(tensor_to_image(frame))
    plt.show()
    plt.figure()


def calculate_similarity_to_start_frames(frame_encoding: torch.Tensor, start_frame_encodings: torch.Tensor) -> float:
    x, y = start_frame_encodings, frame_encoding.repeat(start_frame_encodings.size(0), 1, 1)
    similarity = F.cosine_similarity(x, y, dim=-1)
    # similarity = -torch.norm(x - y, dim=-1)
    # ten_percent = max(int(0.1 * similarity.size(0)), 1)
    # similarity.sort(descending=True)[0][:ten_percent].mean().item()
    return similarity.mean().item()


def display_most_similar_frames(
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
        # sharex='col',
        # sharey='row',
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


def run_experiment():
    device = 'cpu'
    encoder = load_encoder('frame-comparison-experiment/checkpoint_bag_pick_up.pt', device)

    scan_frames = load_video_frames('../data/scan-over-bag-wide2.mp4', cache_path='../cache/scan_frames.pt').to(device)

    start_frames = get_start_frames('../data/bag_pick_up_data', cache_path='../cache/start_frames.pt').to(device)
    display_frame(start_frames[50, :, :, :], title='Sample Start Frame')

    scan_frame_encodings = encode_frames(encoder, scan_frames, cache_path='../cache/scan_frame_encodings.pt')
    start_frame_encodings = encode_frames(encoder, start_frames, cache_path='../cache/start_frame_encodings.pt')

    similarity_to_start_frames = []
    for scan_frame in tqdm(scan_frame_encodings, desc='Comparing each scan frame to start frames'):
        similarity_to_start_frames.append(calculate_similarity_to_start_frames(scan_frame, start_frame_encodings))

    plt.title(f'Similarity to Start Frames')
    plt.xlabel('Frame')
    plt.ylabel('Similarity')
    plt.scatter(range(len(similarity_to_start_frames)), similarity_to_start_frames)
    plt.show()

    display_most_similar_frames(scan_frames, similarity_to_start_frames, top_k=20)


def cache_operation(operation: typing.Callable[[], torch.Tensor], cache_path: str = None):
    if cache_path:
        try:
            return torch.load(cache_path, weights_only=True)
        except FileNotFoundError:
            pass
    data = operation()
    if cache_path:
        torch.save(data, cache_path)
    return data


def get_start_frames(dataset_root: str, cache_path: str) -> torch.Tensor:

    def load_start_frames_from_dataset_videos() -> torch.Tensor:
        files = list(Path(dataset_root).rglob('*.mp4'))
        if not files:
            raise ValueError('No files found in dataset root.')
        start_frames = []
        for i, file in enumerate(tqdm(files, desc=f'Extracting start frames from dataset at {dataset_root}')):
            video_reader = decord.VideoReader(
                str(file),
                ctx=decord.cpu(0),
                width=256,
                height=256,
                num_threads=-1,
            )
            frame = torch.Tensor(video_reader[0].asnumpy())
            start_frames.append(frame)

        start_frames = torch.stack(start_frames) / 255.0
        start_frames = einops.rearrange(start_frames, "t h w c -> t c h w")
        return start_frames

    return cache_operation(load_start_frames_from_dataset_videos, cache_path)


run_experiment()
