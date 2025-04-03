import math
import os
import typing
from pathlib import Path

import decord
import einops
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn
from tqdm import tqdm


# This file was taken from https://github.com/jaron-cui/camera-frame-alignment/blob/main/src/scan_buffer.py
class ScanBuffer:
    def __init__(
        self,
        encoder: nn.Module,
        evaluator: typing.Callable[[torch.Tensor, torch.Tensor], float]
    ):
        self.encoder = encoder
        self.evaluator = evaluator
        self.frames: typing.List[typing.Tuple[torch.Tensor, torch.Tensor]] = []
        self.scan_angles: typing.List[float] = []
        self.scores: typing.List[float] = []
        self.histogram = []  # 'histogram' is a bit of a misnomer - more like 'sliding window'

    def clear(self):
        self.frames.clear()
        self.scan_angles.clear()
        self.scores.clear()
        self.histogram.clear()

    def add_frame(self, rgb: torch.Tensor, depth: torch.Tensor, scan_angle_degrees: float):
        self.frames.append((rgb, depth))
        self.scan_angles.append(normalize_angle_degrees(scan_angle_degrees))

    def process(self):
        # ensure that frames are ordered by scan angle for easier processing
        # sorted_indices = arg_sort(self.scan_angles)
        # self.frames = reorder_list(self.frames, sorted_indices)
        # self.scan_angles = reorder_list(self.scan_angles, sorted_indices)

        # score each frame
        rgb_encodings = batched_encoding(self.encoder, torch.stack([rgb for rgb, _ in self.frames]))
        depth_encodings = batched_encoding(self.encoder, torch.stack([depth for _, depth in self.frames]))

        self.scores = [
            self.evaluator(rgb_encoding, depth_encoding)
            for rgb_encoding, depth_encoding in zip(rgb_encodings, depth_encodings)
        ]

    def retrieve_best_scan_angle(self, angle_tolerance: float = 10, top_proportion: float = 0.2):
        if len(self.frames) != len(self.scores):
            raise RuntimeError(
                'Cannot retrieve best scan angle when frames have not been processed! '
                'Call ScanBuffer.process() first.'
            )

        self.histogram = [0] * len(self.frames)
        ranked_scores = set(list(reversed(arg_sort(self.scores)))[:math.ceil(len(self.scores) * top_proportion)])
        for i in ranked_scores:
            angle = self.scan_angles[i]
            considered = [j for j in ranked_scores if angle_difference(angle, self.scan_angles[j]) <= angle_tolerance]
            # set the processed score for a given frame to the number of proximal top scoring frames
            self.histogram[i] = len(considered)

        return self.scan_angles[arg_max(self.histogram)]


class ResNetHiddenEncoder(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        model = torchvision.models.resnet34(weights=torchvision.models.ResNet34_Weights.IMAGENET1K_V1).eval()
        self.layers = nn.Sequential(*list(model.children())[:-2])

    def forward(self, x):
        return self.layers(x).flatten(start_dim=1)


class SimilarityEvaluator:
    def __init__(self, reference_path: str, rgb_weight: float = 1.0, depth_weight: float = 1.0):
        reference = torch.load(reference_path, weights_only=True)
        self.rgb_reference = reference[0]
        self.depth_reference = reference[1]
        self.rgb_weight = rgb_weight
        self.depth_weight = depth_weight

    def __call__(self, rgb_encoding: torch.Tensor, depth_encoding: torch.Tensor) -> float:
        rgb_score = nn.functional.cosine_similarity(rgb_encoding.unsqueeze(0), self.rgb_reference.unsqueeze(0)).item()
        depth_score = nn.functional.cosine_similarity(depth_encoding.unsqueeze(0),
                                                      self.depth_reference.unsqueeze(0)).item()
        return self.rgb_weight * rgb_score + self.depth_weight * depth_score

    @staticmethod
    def create_reference(
        encoder: nn.Module,
        dataset_path: str,
        is_unprocessed_dataset: bool,
        reference_save_path: str,
        start_frame_count: int = 5,
        skip_every: int = 3
    ):
        if not is_unprocessed_dataset:
            raise NotImplementedError()

        video_paths = [file for file in Path(dataset_path).rglob('*.mp4') if 'Depth' not in str(file.name)]
        depth_folder_paths = [
            next(Path(os.path.dirname(path)).rglob('Depth_Images_*')) for path in video_paths
        ]

        rgb_start_frames = load_start_frames(
            video_paths, transpose=True, count=start_frame_count, skip_every=skip_every)
        depth_start_frames = depth_to_rgb(load_depth_start_frames(
            depth_folder_paths, count=start_frame_count, skip_every=skip_every), resize=(256, 256))

        rgb_encodings = batched_encoding(encoder, rgb_start_frames, verbose=True)
        depth_encodings = batched_encoding(encoder, 1 - 0.5 * depth_start_frames, verbose=True)

        reference = torch.stack([
            rgb_encodings.mean(dim=0).squeeze(0),
            depth_encodings.mean(dim=0).squeeze(0)
        ])

        torch.save(reference, reference_save_path)


def load_dataset_start_frames(
    dataset_root: str,
    transpose: bool = False,
    count: int = 1,
    skip_every: int = 1
) -> torch.Tensor:
    """
    Taken from camera-frame-alignment/video.py/load_dataset_start_frames

    :param dataset_root:
    :param transpose:
    :param count:
    :param skip_every:
    :return:
    """
    file_paths = [file for file in Path(dataset_root).rglob('*.mp4') if 'Depth' not in str(file.name)]
    return load_start_frames(file_paths, transpose, count, skip_every)


def load_start_frames(
    file_paths: typing.List[Path],
    transpose: bool = False,
    count: int = 1,
    skip_every: int = 1
):
    """
    Taken from camera-frame-alignment/video.py/load_start_frames

    :param file_paths:
    :param transpose:
    :param count:
    :param skip_every:
    :return:
    """
    if not file_paths:
        raise ValueError('Must provide at least one file from which to load start frames.')
    start_frames = torch.zeros((len(file_paths) * count, 256, 256, 3))
    for i, file in enumerate(tqdm(file_paths, desc=f'Extracting start frames from {len(file_paths)} files')):
        video_reader = decord.VideoReader(
            str(file),
            ctx=decord.cpu(0),
            width=256,
            height=256,
            num_threads=-1,
        )
        for j, frame_index in enumerate(range(count)):
            frame = torch.Tensor(video_reader[frame_index * skip_every].asnumpy())
            start_frames[i * count + j] = frame

    start_frames.divide_(255.0)
    start_frames = einops.rearrange(start_frames, 't h w c -> t c h w')
    if transpose:
        start_frames = start_frames.transpose(-1, -2).flip(dims=[-2])
    return start_frames


def load_depth_start_frames(
    folder_paths: typing.List[Path],
    count: int = 1,
    skip_every: int = 1
) -> torch.Tensor:
    """
    Loads the start frames from each file path in the same order as given.
    Every frame must be of the same width and height.

    Taken from camera-frame-alignment/video.py/load_depth_start_frames.

    :param file_paths: a list of file paths
    :param count:
    :param skip_every:
    :param cache_path:
    :return: a numpy array of shape ((file_count*count) x height x width)
    """
    frames = []
    num_folders = len(folder_paths)
    for folder_path in tqdm(folder_paths, desc=f'Loading depth start frames from {num_folders} folders'):
        file_paths = list(folder_path.glob('*.bin'))

        # order the depth files by frame number: depth bin files are named 0.pt, 1.pt, ..., 11.pt, etc..
        file_name_indices = [int(os.path.basename(path).split('.')[0]) for path in file_paths]
        file_path_ordering = sorted(range(len(file_name_indices)), key=lambda i: file_name_indices[i])
        file_paths = reorder_list(file_paths, file_path_ordering)

        frame_indices = [number * skip_every for number in range(count)]
        if len(file_paths) < max(frame_indices):
            raise ValueError(f'Too few depth files to reach count in {folder_path}:'
                             f'{", ".join([p.name for p in file_paths])}')

        file_frames = load_depth_frames_from_individual_binaries([file_paths[i] for i in frame_indices])
        frames.extend(file_frames)
    return torch.from_numpy(np.array(frames))


def load_depth_frames_from_individual_binaries(
    file_paths: typing.List[Path],
    width: int = 256,
    height: int = 192
) -> torch.Tensor:
    """
    Taken from camera-frame-alignment/video.py/load_depth_frames_from_individual_binaries.

    :param file_paths:
    :param width:
    :param height:
    :return:
    """
    frames = []
    for file_path in file_paths:
        depth_map = np.fromfile(file_path, dtype=np.float32).reshape((height, width))
        frames.append(depth_map)
    return torch.from_numpy(np.stack(frames))


def batched_encoding(
    encoder: nn.Module,
    frames: torch.Tensor,
    batch_size: int = 32,
    verbose: bool = False
) -> torch.Tensor:
    """
    Encodes a large sequence of RGB images in subdivisions to accommodate memory constraints.

    Taken from camera-frame-alignment/video.py/encode_frames.

    :param encoder: an image encoder that accepts inputs of shape (batch x channel x height x width)
    :param frames: a sequence of RGB images of shape (count x channel x height x width)
    :param batch_size: the size of the subdivisions to be encoded
    :param verbose: display a tqdm progress bar
    :return: the image encodings in the same order as the images were given
    """
    if torch.cuda.is_available():
        encoder = encoder.cuda()
        def prep(t): return t.cuda()
    else:
        def prep(t): return t

    dataset = torch.utils.data.TensorDataset(frames)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    if verbose:
        dataloader = tqdm(dataloader, desc=f'Encoding frames in batches of {batch_size}')

    outputs = torch.zeros((frames.size(0), *encoder(prep(frames[0].unsqueeze(0))).shape[1:]))
    for i, (batch,) in enumerate(dataloader):
        output = encoder(prep(batch)).detach()
        outputs[i * batch_size:(i + 1) * batch_size] = output
    return outputs


def depth_to_rgb(depth: torch.Tensor, resize: typing.Tuple[int, int] = None, max_depth: int = 5) -> torch.Tensor:
    depth = depth.clamp_max(max_depth).divide(max_depth)
    rgb = depth.unsqueeze(1).expand(-1, 3, -1, -1)
    if resize is not None:
        resize = transforms.Resize((256, 256))
        rgb = resize(rgb)
    return rgb


def normalize_angle_degrees(angle_degrees: float):
    mod = angle_degrees % 360
    return mod if mod >= 0 else mod + 360


def angle_difference(angle1: float, angle2: float):
    raw_diff = normalize_angle_degrees(angle2 - angle1)
    return min(raw_diff, 360 - raw_diff)


def arg_sort(x: typing.List[typing.Any]) -> typing.List[int]:
    return sorted(range(len(x)), key=lambda i: x[i])


def arg_max(x: typing.List[float]) -> int:
    return max(range(len(x)), key=lambda i: x[i])


T = typing.TypeVar('T')


def reorder_list(x: typing.List[T], indices: typing.List[int]) -> typing.List[T]:
    return [x[i] for i in indices]
