import os
import typing
from pathlib import Path

import einops
import decord
import liblzfse
import numpy as np
import torch
import torchvision.transforms as transforms
from tqdm import tqdm

from utils import cache_operation, PathLike


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


def load_dataset_start_frames(
    dataset_root: str,
    cache_path: str = None,
    transpose: bool = False,
    count: int = 1,
    skip_every: int = 1
) -> torch.Tensor:
    file_paths = [file for file in Path(dataset_root).rglob('*.mp4') if 'Depth' not in str(file.name)]
    return load_start_frames(file_paths, cache_path, transpose, count, skip_every)


def load_start_frames(
    file_paths: typing.Iterable[PathLike],
    cache_path: PathLike = None,
    transpose: bool = False,
    count: int = 1,
    skip_every: int = 1
) -> torch.Tensor:

    def operation() -> torch.Tensor:
        if not file_paths:
            raise ValueError('Must provide at least one file from which to load start frames.')
        start_frames = []
        for i, file in enumerate(tqdm(file_paths, desc=f'Extracting start frames from {len(file_paths)} files')):
            video_reader = decord.VideoReader(
                str(file),
                ctx=decord.cpu(0),
                width=256,
                height=256,
                num_threads=-1,
            )
            for frame_index in range(count):
                frame = torch.Tensor(video_reader[frame_index * skip_every].asnumpy())
                start_frames.append(frame)

        start_frames = torch.stack(start_frames) / 255.0
        start_frames = einops.rearrange(start_frames, 't h w c -> t c h w')
        if transpose:
            start_frames = start_frames.transpose(-1, -2).flip(dims=[-2])
        return start_frames

    return cache_operation(operation, cache_path)


def load_depth_frames_from_video_archive(file_path: PathLike, cache_path: PathLike = None) -> torch.Tensor:
    """
    Loads video depth data from a binary file containing all depth frames.

    :param file_path: the path to a binary file containing depth data
    :param cache_path:
    :return: a numpy array of shape (frame_count x height x width)
    """
    def operation() -> torch.Tensor:
        nonlocal file_path
        if not isinstance(file_path, Path):
            file_path = Path(file_path)
        depth_data = np.frombuffer(
            liblzfse.decompress(file_path.read_bytes()), dtype=np.float32
        )
        depth_data = depth_data.reshape((-1, 192, 256))

        return torch.tensor(depth_data)

    return cache_operation(operation, cache_path)


def load_depth_frames_from_individual_binaries(
    folder_path: PathLike,
    width: int = 256,
    height: int = 192,
    cache_path: PathLike = None
) -> torch.Tensor:
    def operation() -> torch.Tensor:
        nonlocal folder_path
        if not isinstance(folder_path, Path):
            folder_path = Path(folder_path)
        file_paths = sorted(folder_path.glob('*.bin'), key=lambda path: os.path.basename(path))
        frames = []
        for file_path in tqdm(file_paths, desc=f'Loading depth frames from {folder_path}'):
            depth_map = np.fromfile(file_path, dtype=np.float32).reshape((height, width))
            frames.append(depth_map)
        return torch.from_numpy(np.stack(frames))

    return cache_operation(operation, cache_path)


def load_depth_start_frames(
    file_paths: typing.Iterable[PathLike],
    count: int = 1,
    skip_every: int = 1,
    cache_path: PathLike = None
) -> torch.Tensor:
    """
    Loads the start frames from each file path in the same order as given.
    Every frame must be of the same width and height.

    :param file_paths: a list of file paths
    :param count:
    :param skip_every:
    :param cache_path:
    :return: a numpy array of shape ((file_count*count) x height x width)
    """
    def operation() -> torch.Tensor:
        frames = []
        num_files = len(list(file_paths))
        for file_path in tqdm(file_paths, desc=f'Extracting depth start frames from {num_files} files.'):
            file_frames = load_depth_frames_from_video_archive(file_path)
            frame_indices = [number * skip_every for number in range(count)]
            frames.extend(file_frames[frame_indices, :, :])
        return torch.from_numpy(np.array(frames))

    return cache_operation(operation, cache_path)


def depth_to_rgb(depth: torch.Tensor, resize: typing.Tuple[int, int] = None, max_depth: int = 5) -> torch.Tensor:
    depth = depth.clamp_max(max_depth).divide(max_depth)
    rgb = depth.unsqueeze(1).expand(-1, 3, -1, -1)
    if resize is not None:
        resize = transforms.Resize((256, 256))
        rgb = resize(rgb)
    return rgb


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
