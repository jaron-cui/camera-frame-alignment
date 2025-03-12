from pathlib import Path

import cv2
import decord
import einops
import torch
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision
from sklearn.cluster import DBSCAN
from torch import nn
from tqdm import tqdm


def load_encoder(weight_path: str, device) -> nn.Module:
    # model = torchvision.models.resnet34(weights=torchvision.models.ResNet34_Weights.IMAGENET1K_V1)
    # return model.to(device)
    class CustomModel(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            model = torchvision.models.resnet34(weights=torchvision.models.ResNet34_Weights.IMAGENET1K_V1)
            self.layers = nn.Sequential(*list(model.children())[:-2])

        def forward(self, x):
            return self.layers(x).flatten(start_dim=1)

    return CustomModel().to(device)
    # # Taken from imitation-in-homes/run.py line 70: _init_model()
    # print('Loading encoder.')
    # model = hydra.utils.instantiate(DictConfig({
    #     '_target_': 'models.encoders.timm_encoders.TimmSSL',
    #     'model_name': 'hf-hub:notmahi/dobb-e'
    # })).to(device)
    #
    # checkpoint = torch.load(weight_path, device)
    # model.load_state_dict(checkpoint["model"])
    # return model


def load_video_frames(mp4_path: str, refresh: bool = True) -> torch.Tensor:
    # Taken from data-collection/utils/new_gripper_model.py line 54
    # print(f'Loading frames from {mp4_path}.')
    cache_file = Path(os.path.dirname(mp4_path)) / 'mp4_cache.pt'
    if not refresh:
        try:
            return torch.load(cache_file, weights_only=True)
        except FileNotFoundError:
            pass

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
    torch.save(frames_tensor, cache_file)
    return frames_tensor


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
    if cache_path:
        try:
            return torch.load(cache_path, weights_only=True)
        except FileNotFoundError:
            pass
    dataset = torch.utils.data.TensorDataset(frames)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    outputs = []
    for batch, in dataloader:
        output = encoder.forward(batch).detach().squeeze(0).unsqueeze(1)
        outputs.append(output)
    encoded = torch.cat(outputs, dim=0)
    if cache_path:
        torch.save(encoded, cache_path)
    return encoded


# def extract_best_frames(similarity: torch.Tensor, count: int) -> list[int]:
#     data = similarity.reshape(-1, 1)
#
#     # Step 1: Cluster the data to identify peaks
#     dbscan = DBSCAN(eps=0.003, min_samples=5)  # Tune eps based on spread
#     labels = dbscan.fit_predict(data)
#     # print(labels)
#     # Step 2: Find the local maxima within each cluster
#     unique_labels = set(labels)
#     peak_points = []
#
#     for label in unique_labels:
#         if label == -1:  # Ignore noise
#             continue
#         cluster_points = data[labels == label]
#         max_point = np.argmax(cluster_points[:, 0])  # Highest y-value in cluster
#         peak_points.append(int(max_point))
#         print('l', label)
#
#     # Step 3: Sort the peaks by height and select the top 10
#     peak_points = sorted(peak_points, key=lambda p: similarity[p].item(), reverse=True)[:count]
#     # peak_points = torch.from_numpy(np.array(peak_points))
#     # print(peak_points.shape, peak_points)
#     return peak_points


def run_experiment():
    device = 'cpu'
    encoder = load_encoder('frame-comparison-experiment/checkpoint_bag_pick_up.pt', device)

    scan_frames = load_video_frames('frame-comparison-experiment/scan-over-bag-wide2.mp4', refresh=False).to(device)
    def show_scan_frame(index: int):
        plt.title(f'Scan View Frame #{index}')
        plt.imshow(tensor_to_image(scan_frames[index, :, :, :]))
        plt.show()
        plt.figure()

    # selected_scan_frame = 160
    # for index in [selected_scan_frame, 70, 140]:
    #     show_scan_frame(index)
    start_frames = get_start_frames('/home/jaron_cui/Desktop/data/bag_pick_up_data', refresh=False).to(device)
    plt.title('Sample Start Frame')
    plt.imshow(tensor_to_image(start_frames[50, :, :, :]))
    plt.show()
    plt.figure()
    # ref_frames = tensors_to_image(start_frames).squeeze(0)
    # query_frames = tensors_to_image(scan_frames).squeeze(0)
    print('Encoding scan frames.')
    scan_frame_encodings = encode_frames(encoder, scan_frames, cache_path='scan_frames.pt')
    print('Encoding start frames.')
    start_frame_encodings = encode_frames(encoder, start_frames, cache_path='start_frames.pt')
    print(scan_frame_encodings.shape)
    # reference_descriptors = []
    # for ref in ref_frames:
    #     reference_descriptors.append(extract_orb_features(ref)[1])

    start_frame_similarity = []
    for scan_frame in tqdm(scan_frame_encodings, desc='Comparing each scan frame to start frames'):
        # scan_frame = scan_frame.unsqueeze(0)
        # print(start_frame_encodings.shape, scan_frame.repeat(start_frame_encodings.size(0), 1, 1).shape, scan_frame.shape)
        x, y = start_frame_encodings, scan_frame.repeat(start_frame_encodings.size(0), 1, 1)
        similarity = F.cosine_similarity(x, y, dim=-1)
        # similarity = -torch.norm(x - y, dim=-1)
        # similarity = torch.zeros(ref_frames.size)
        # query = extract_orb_features(scan_frame)[1]
        # best_matches = 0
        # for i, ref in enumerate(reference_descriptors):
        #     matches = match_features(query, ref)
        #     similarity[i] = len(matches)
        #     # best_matches = max(matches, best_matches)

        # print(similarity.shape, similarity.__class__)
        ten_percent = max(int(0.1 * similarity.size(0)), 1)
        # start_frame_similarity.append(similarity.sort(descending=True)[0][:ten_percent].mean().item())
        start_frame_similarity.append(similarity.mean().item())

    plt.title(f'Similarity to Start Frames')
    plt.xlabel('Frame')
    plt.ylabel('Similarity')
    plt.scatter(range(len(start_frame_similarity)), start_frame_similarity)
    plt.show()

    _, best_matches = torch.Tensor(start_frame_similarity).sort(descending=True)
    # best_match = best_matches[0]
    for match in best_matches[:10]:
        show_scan_frame(match)

    # frame_encoding = scan_frame_encodings[best_match].unsqueeze(0)
    # # print(scan_frame_encodings.shape, frame_encoding.shape)
    # scan_similarity = F.cosine_similarity(
    #     scan_frame_encodings, frame_encoding.repeat(scan_frame_encodings.size(0), 1, 1), dim=-1).detach().numpy()
    # plt.title(f'Similarity Within Scan #{best_match}')
    # plt.xlabel('Frame')
    # plt.ylabel('Similarity')
    # plt.plot(range(len(scan_similarity)), scan_similarity)
    # plt.figure()


def get_start_frames(dataset_root: str, refresh: bool = True) -> torch.Tensor:
    cache_file = Path(dataset_root) / 'start_frames.pt'
    if not refresh:
        try:
            return torch.load(cache_file, weights_only=True)
        except FileNotFoundError:
            pass
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
    torch.save(start_frames, cache_file)
    return start_frames

run_experiment()