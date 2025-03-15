import typing
from pathlib import Path

import einops
import decord
import torch
from tqdm import tqdm

from utils import cache_operation


def load_video_frames(mp4_path: str, cache_path: str = None, skip_every: int = 1) -> torch.Tensor:
    if skip_every < 1:
        raise ValueError('skip_every must be at least 1.')

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
            if i % skip_every != 0:
                continue
            frames.append(torch.Tensor(frame.asnumpy()))

        frames_tensor = torch.stack(frames)
        frames_tensor = frames_tensor / 255.0
        frames_tensor = einops.rearrange(frames_tensor, "t h w c -> t c w h").flip(dims=[2])
        # frames_tensor = NORMALIZER(frames_tensor)
        return frames_tensor

    return cache_operation(extract_video_frames, cache_path=cache_path)


def load_start_frames(dataset_root: str, cache_path: str = None, transpose: bool = False, count: int = 1) -> torch.Tensor:

    def load_start_frames_from_dataset_videos() -> torch.Tensor:
        files = [file for file in Path(dataset_root).rglob('*.mp4') if 'Depth' not in str(file.name)]
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
            for frame_index in range(count):
                frame = torch.Tensor(video_reader[frame_index].asnumpy())
                start_frames.append(frame)

        start_frames = torch.stack(start_frames) / 255.0
        start_frames = einops.rearrange(start_frames, 't h w c -> t c h w')
        if transpose:
            start_frames = start_frames.transpose(-1, -2).flip(dims=[-2])
        return start_frames

    return cache_operation(load_start_frames_from_dataset_videos, cache_path)


def encode_frames(
    encoder: typing.Callable[[torch.Tensor], torch.Tensor],
    frames: torch.Tensor,
    cache_path: str = None,
    batch_size: int = 128
) -> torch.Tensor:
    """
    Expects frames of shape (batch x height x width x channels) and returns
    encodings of shape (batch x encoding_length)
    """
    def encode() -> torch.Tensor:
        dataset = torch.utils.data.TensorDataset(frames)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
        outputs = []
        for batch, in tqdm(dataloader, desc=f'Encoding frames in batches of {batch_size}'):
            output = encoder(batch).detach()
            outputs.append(output)
        return torch.cat(outputs, dim=0)

    return cache_operation(encode, cache_path=cache_path)
