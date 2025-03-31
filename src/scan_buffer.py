import math
import typing
from pathlib import Path

import decord
import einops
import torch
import torchvision
from torch import nn
from tqdm import tqdm


class ScanBuffer:
    def __init__(
        self,
        encoder: typing.Callable[[torch.Tensor], torch.Tensor],
        evaluator: typing.Callable[[torch.Tensor], float]
    ):
        self.encoder = encoder
        self.evaluator = evaluator
        self.frames: typing.List[torch.Tensor] = []
        self.scan_angles: typing.List[float] = []
        self.scores: typing.List[float] = []
        self.histogram = []  # 'histogram' is a bit of a misnomer - more like 'sliding window'

    def clear(self):
        self.frames.clear()
        self.scan_angles.clear()
        self.scores.clear()
        self.histogram.clear()

    def add_frame(self, image: torch.Tensor, scan_angle_degrees: float):
        self.frames.append(image)
        self.scan_angles.append(normalize_angle_degrees(scan_angle_degrees))

    def process(self):
        # ensure that frames are ordered by scan angle for easier processing
        # sorted_indices = arg_sort(self.scan_angles)
        # self.frames = reorder_list(self.frames, sorted_indices)
        # self.scan_angles = reorder_list(self.scan_angles, sorted_indices)

        # score each frame
        encodings = batched_encoding(self.encoder, self.frames)
        self.scores = [self.evaluator(encoding) for encoding in encodings]

    def retrieve_best_scan_angle(self, angle_tolerance: float = 10, top_proportion: float = 0.2):
        if len(self.frames) != len(self.scores):
            raise RuntimeError(
                'Cannot retrieve best scan angle when frames have not been processed! '
                'Call ScanBuffer.process() first.'
            )
        self.histogram = [0] * len(self.frames)
        ranked_scores = set(reversed(arg_sort(self.scores))[:math.ceil(len(self.scores) * top_proportion)])
        for i in ranked_scores:
            angle = self.scan_angles[i]
            considered = [j for j in ranked_scores if angle_difference(angle, self.scan_angles[j]) <= angle_tolerance]
            # set the processed score for a given frame to the number of proximal top scoring frames
            self.histogram[i] = len(considered)
        return self.scan_angles[arg_max(self.histogram)]


class ResNetHiddenEncoder:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        model = torchvision.models.resnet34(weights=torchvision.models.ResNet34_Weights.IMAGENET1K_V1).eval()
        self.layers = nn.Sequential(*list(model.children())[:-2])

    def forward(self, x):
        return self.layers(x).flatten(start_dim=1)


class SimilarityEvaluator:
    def __init__(self, reference_path: str):
        self.reference = torch.load(reference_path, weights_only=True)

    def __call__(self, encoding: torch.Tensor) -> float:
        return nn.functional.cosine_similarity(encoding, self.reference).item()

    @staticmethod
    def create_reference(
        encoder: typing.Callable[[torch.Tensor], torch.Tensor],
        training_demo_dataset_path: str,
        reference_save_path: str
    ):
        start_frames = load_dataset_start_frames(training_demo_dataset_path)
        # TODO: implement


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

def batched_encoding(
    encoder: typing.Callable[[torch.Tensor], torch.Tensor],
    frames: typing.List[torch.Tensor],
    batch_size: int = 32
) -> torch.Tensor:
    """
    Taken from camera-frame-alignment/video.py/encode_frames.

    :param encoder:
    :param frames:
    :param batch_size:
    :return:
    """
    dataset = torch.utils.data.TensorDataset(torch.stack(frames))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    outputs = []
    for batch, in dataloader:
        output = encoder(batch).detach()
        outputs.append(output)
    return torch.cat(outputs, dim=0)


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
