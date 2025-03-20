import typing

import torch
import torch.nn.functional as F
from tqdm import tqdm

import visualizations
import encoders
import video

# This code has all been moved to experiments.ipynb. Go there, instead.


def average_cosine_similarity(frame_encoding: torch.Tensor, start_frame_encodings: torch.Tensor) -> float:
    x, y = start_frame_encodings, frame_encoding.unsqueeze(0).expand(start_frame_encodings.size(0), -1)
    similarity = F.cosine_similarity(x, y, dim=-1)
    return similarity.mean().item()


def max_cosine_similarity(frame_encoding: torch.Tensor, start_frame_encodings: torch.Tensor) -> float:
    x, y = start_frame_encodings, frame_encoding.unsqueeze(0).expand(start_frame_encodings.size(0), -1)
    similarity = F.cosine_similarity(x, y, dim=-1)
    return similarity.max().item()


def run_experiment():
    device = 'cuda'
    # encoder = load_resnet_encoder('frame-comparison-experiment/checkpoint_bag_pick_up.pt', device)
    # encoder = load_dino_hidden_state_encoder(device)
    # encoder = load_dino_cls_encoder(device)
    encoder = encoders.clip_encoder(device)

    scan_frames = video.load_video_frames('../scans/scan-over-bag-wide2.mp4', cache_path='../cache/scan_frames.pt').to(device)
    start_frames = video.load_dataset_start_frames('../datasets/bag_pick_up_data', cache_path='../cache/start_frames.pt').to(device)

    alignment_scores = compute_alignment_scores(
        scan_frames,
        start_frames,
        encoder,
        average_cosine_similarity,
        # scan_frame_encodings_cache_path='../cache/scan_frame_encodings.pt',
        # target_frame_encodings_cache_path='../cache/start_frame_encodings.pt'
    )

    visualizations.display_frame(start_frames[80, :, :, :], title='Sample Start Frame')
    visualizations.display_alignment_scores(alignment_scores)
    visualizations.display_most_aligned_frames(scan_frames, alignment_scores, top_k=20)


def compute_alignment_scores(
    scan_frames: torch.Tensor,
    target_frames: torch.Tensor,
    encoder: typing.Callable[[torch.Tensor], torch.Tensor],
    compute_encoding_alignment_score: typing.Callable[[torch.Tensor, torch.Tensor], float],
    scan_frame_encodings_cache_path: str = None,
    target_frame_encodings_cache_path: str = None
):
    scan_frame_encodings = video.encode_frames(
        encoder,
        scan_frames,
        cache_path=scan_frame_encodings_cache_path,
        batch_size=32
    )
    target_frame_encodings = video.encode_frames(
        encoder,
        target_frames,
        cache_path=target_frame_encodings_cache_path,
        batch_size=32
    )

    alignment_scores = []
    for scan_frame in tqdm(scan_frame_encodings, desc='Comparing each scan frame to start frames'):
        alignment_scores.append(compute_encoding_alignment_score(scan_frame, target_frame_encodings))

    return alignment_scores


if __name__ == '__main__':
    run_experiment()
