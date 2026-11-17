"""Collect source-aligned face patches for reproducible appearance and detail review."""

from __future__ import annotations

import hashlib

import cv2
import numpy as np

from app.detectors.image.faces import detect_faces, iou
from app.evidence.face_detail_binding import SCHEDULE

SIZE = 168
GRID = SIZE // 14
LAYERS = (3, 6)
CHANNELS = 24


def align(pixels, eyes):
    eye, other = sorted(eyes)
    dx, dy = other[0] - eye[0], other[1] - eye[1]
    norm = dx * dx + dy * dy
    if norm < 18 ** 2:
        return None
    r, s = (.32 * SIZE * value / norm for value in (dx, dy))
    transform = np.array([[r, s, .34 * SIZE - r * eye[0] - s * eye[1]],
                          [-s, r, .38 * SIZE + s * eye[0] - r * eye[1]]])
    backward = cv2.invertAffineTransform(transform)
    region = np.array([[.10, .14], [.90, .14], [.10, .88], [.90, .88]]) * SIZE
    source = region @ backward[:, :2].T + backward[:, 2]
    if (source.min(axis=0) < 1).any() or (source.max(axis=0) >= np.asarray(pixels.shape[1::-1]) - 1).any():
        return None
    return cv2.warpAffine(pixels, transform, (SIZE, SIZE), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)


def tracked_faces(store, detector, *, check=None):
    """Choose the longest consecutive run on the same supported face."""
    groups, span, previous = [], [], None
    # The store can supply cached frames before earlier frames that still need decoding.
    for index, pixels in sorted(store.get_many(SCHEDULE).items()):
        if check is not None:
            check()
        if not store.native_pts_verified(index):
            raise ValueError("A sampled face frame lacks a verified decoder position")
        faces = detect_faces(cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR), detector=detector, min_size=56)
        face = faces[0] if faces else None
        eyes = face.landmarks[:2] if face is not None else []
        crop = align(pixels, eyes) if len(eyes) == 2 else None
        if crop is None:
            if span:
                groups.append(span)
            span, previous = [], None
            continue
        if span and (index - span[-1][0] != 6 or iou(face, previous) < .25):
            groups.append(span)
            span = []
        span.append((index, store.table.pts[index], face.box(), crop, hashlib.sha256(pixels.tobytes()).hexdigest(), eyes))
        previous = face
    return max((*groups, span), key=len)


class PatchEncoder:
    def __init__(self, model, device):
        self.model, self.device = model, device
        rng = np.random.default_rng(621112)
        self.projection = [np.linalg.qr(rng.choice((-1., 1.), (384, CHANNELS)))[0].astype(np.float32) for _ in LAYERS]

    def __call__(self, crops):
        import torch

        from app.detectors.video.tracking import DinoEmbedder

        images = np.stack(crops).astype(np.float32) / 255.
        images = (images - DinoEmbedder.MEAN) / DinoEmbedder.STD
        values = []
        with torch.inference_mode():
            for group in np.array_split(images, (len(images) + 7) // 8):
                batch = torch.from_numpy(np.ascontiguousarray(group.transpose(0, 3, 1, 2))).to(self.device)
                response = self.model(pixel_values=batch, output_hidden_states=True, return_dict=True)
                channels = []
                for depth, projection in zip(LAYERS, self.projection, strict=True):
                    patches = response.hidden_states[depth][:, 1:].cpu().numpy()
                    if patches.shape[1:] != (GRID * GRID, 384):
                        raise ValueError("The pinned encoder has a different patch geometry")
                    patches = patches - patches.mean(axis=-1, keepdims=True)
                    patches /= np.maximum(np.linalg.norm(patches, axis=-1, keepdims=True), 1e-6)
                    channels.append((patches @ projection).reshape(-1, GRID, GRID, CHANNELS))
                values.append(np.concatenate(channels, axis=-1))
        return np.concatenate(values)


def regions():
    u = (np.arange(GRID) + .5) / GRID
    x, y = np.meshgrid(u, u)
    distance = ((x - .5) / .35) ** 2 + ((y - .53) / .37) ** 2
    eligible = (x >= .10) & (x <= .90) & (y >= .14) & (y <= .88)
    return (distance <= .61) & eligible, (distance > .61) & (distance <= 1.16) & eligible, (distance > 1.16) & eligible


def appearance(values):
    return np.concatenate([np.median(values[:, zone].mean(axis=1), axis=0) for zone in regions()]).tolist()
