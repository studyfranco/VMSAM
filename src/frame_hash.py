'''Perceptual frame hashes (imagehash.phash), distances and lag alignment.

Distances are fractions of the hash bits: 0 identical, about 0.5 unrelated.
pHash is unstable on near-black or near-white frames, so those are compared on a
small grey thumbnail instead (`pixel_distance`).
'''

import numpy as np
import imagehash
from PIL import Image

HASH_SIZE = 32
HASH_BITS = HASH_SIZE * HASH_SIZE
COLOUR_HASH_SIZE = 16
FRAME_WIDTH = 160
FRAME_HEIGHT = 90

# Luma std under which a frame is flat and never takes part in an alignment.
FLAT_STD = 3.0
COLOUR_WEIGHT = 0.5
# An alignment is kept only when its minimum beats every lag two or more frames away by this.
ALIGN_MARGIN = 0.01
ALIGN_MIN_FRAMES = 6
# Same content: mean content distance of a 6+ frame window at most this
# (same content up to 0.278, other content from 0.457).
SAME_CONTENT_MAX = 0.37
SAME_FRAME_MAX = 0.07

# Near-uniform frames: share of pixels <= DARK_LEVEL or >= LIGHT_LEVEL.
DARK_LEVEL = 48
LIGHT_LEVEL = 207
UNIFORM_FRACTION = 0.75
THUMB_FRACTION = 0.6
THUMB_WIDTH = 32
THUMB_HEIGHT = 18
PIXEL_STD_FLOOR = 12.0
SAME_PIXEL_MAX = 0.005


class FrameHashes:
    '''Hashes of a run of frames: packed grey and Cb/Cr pHash, luma std, near-uniform
    fraction and thumbnails. Indexing returns another FrameHashes.'''

    __slots__ = ("grey", "colour", "std", "extreme", "thumb_index", "thumbs")

    def __init__(self, grey, colour, std, extreme, thumb_index, thumbs):
        self.grey = np.asarray(grey, dtype=np.uint64)
        self.colour = None if colour is None else np.asarray(colour, dtype=np.uint64)
        self.std = np.asarray(std, dtype=np.float32)
        self.extreme = np.asarray(extreme, dtype=np.float32)
        self.thumb_index = np.asarray(thumb_index, dtype=np.int32)
        self.thumbs = np.asarray(thumbs, dtype=np.uint8)

    def __len__(self):
        return len(self.grey)

    def __getitem__(self, key):
        return FrameHashes(self.grey[key], None if self.colour is None else self.colour[key],
                           self.std[key], self.extreme[key], self.thumb_index[key], self.thumbs)

    @property
    def bits(self):
        return self.grey.shape[-1] * 64


def frames_from_raw(blob, width, height, channels=3):
    '''Raw rgb24 bytes to a (n, h, w, 3) uint8 array; a trailing partial frame is dropped.'''
    size = width * height * channels
    count = len(blob) // size
    return np.frombuffer(blob[:count * size], dtype=np.uint8).reshape(count, height, width, channels)


def _pack(image, hash_size):
    return np.packbits(np.asarray(imagehash.phash(image, hash_size=hash_size).hash,
                                  dtype=bool).ravel()).view(np.uint64)


def hash_frames(frames, colour=False):
    '''Hash (n, h, w, 3) RGB frames; `colour` also hashes the Cb and Cr planes.'''
    grey, chroma, std, extreme, thumb_index, thumbs = [], [], [], [], [], []
    for frame in np.asarray(frames):
        img = Image.fromarray(np.ascontiguousarray(frame), "RGB")
        luma = img.convert("L")
        grey.append(_pack(luma, HASH_SIZE))
        pixels = np.asarray(luma)
        std.append(pixels.astype(np.float32).std())
        extreme.append(max(np.count_nonzero(pixels <= DARK_LEVEL),
                           np.count_nonzero(pixels >= LIGHT_LEVEL)) / pixels.size)
        if extreme[-1] >= THUMB_FRACTION:
            thumb_index.append(len(thumbs))
            thumbs.append(np.asarray(luma.resize((THUMB_WIDTH, THUMB_HEIGHT), Image.BOX), np.uint8))
        else:
            thumb_index.append(-1)
        if colour:
            _, cb, cr = img.convert("YCbCr").split()
            chroma.append([_pack(cb, COLOUR_HASH_SIZE), _pack(cr, COLOUR_HASH_SIZE)])
    n = len(grey)
    return FrameHashes(np.stack(grey) if n else np.zeros((0, HASH_BITS // 64), np.uint64),
                       np.asarray(chroma, np.uint64).reshape(n, 2, -1) if colour else None,
                       std, extreme, thumb_index,
                       np.stack(thumbs) if thumbs else np.zeros((0, THUMB_HEIGHT, THUMB_WIDTH)))


def hamming(a, b):
    '''Differing bits between packed hashes (words on the last axis); broadcasts.'''
    return np.bitwise_count(np.bitwise_xor(a, b)).sum(axis=-1, dtype=np.int64)


def pixel_distance(a, b):
    '''(1 - correlation) / 2 of z-scored grey thumbnails: 0 same, about 0.5 unrelated.'''
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    axes = (-2, -1)
    za = (a - a.mean(axis=axes, keepdims=True)) / np.maximum(a.std(axis=axes, keepdims=True), PIXEL_STD_FLOOR)
    zb = (b - b.mean(axis=axes, keepdims=True)) / np.maximum(b.std(axis=axes, keepdims=True), PIXEL_STD_FLOOR)
    return np.square(za - zb).mean(axis=axes) / 4.0


def _pixel_rows(a_extreme, a_index, b_extreme, b_index):
    '''Pairs decided by thumbnails: either side near-uniform, both carrying one.'''
    return (((a_extreme >= UNIFORM_FRACTION) | (b_extreme >= UNIFORM_FRACTION))
            & (a_index >= 0) & (b_index >= 0))


def _pixel_as_bits(pixel):
    '''A pixel distance on the pHash scale: SAME_PIXEL_MAX lands on SAME_FRAME_MAX, 0.5 on 0.5.'''
    pixel = np.asarray(pixel, dtype=np.float64)
    low = pixel * (SAME_FRAME_MAX / SAME_PIXEL_MAX)
    high = SAME_FRAME_MAX + (pixel - SAME_PIXEL_MAX) * ((0.5 - SAME_FRAME_MAX) / (0.5 - SAME_PIXEL_MAX))
    return np.minimum(np.where(pixel <= SAME_PIXEL_MAX, low, high), 0.5)


def content_distance(a, b):
    '''Per-frame (grey + COLOUR_WEIGHT x Cb/Cr) distance of two equal-length FrameHashes.'''
    grey = hamming(a.grey, b.grey) / a.bits
    chroma = hamming(a.colour, b.colour).mean(axis=-1) / (a.colour.shape[-1] * 64)
    content = (grey + COLOUR_WEIGHT * chroma) / (1.0 + COLOUR_WEIGHT)
    rows = _pixel_rows(a.extreme, a.thumb_index, b.extreme, b.thumb_index)
    if rows.any():
        content = np.asarray(content, dtype=np.float64)
        content[rows] = _pixel_as_bits(pixel_distance(a.thumbs[a.thumb_index[rows]],
                                                      b.thumbs[b.thumb_index[rows]]))
    return content


def window_content_distance(ref, other, offset=0):
    '''Mean `content_distance` of `ref` against `other[offset:]`, flat frames left out; NaN
    when the window leaves `other`.'''
    if offset < 0 or offset + len(ref) > len(other):
        return float("nan")
    rows = np.nonzero(ref.std >= FLAT_STD)[0]
    if len(rows) == 0:
        return float("nan")
    return float(np.mean(content_distance(ref[rows], other[rows + offset])))


class Alignment:
    '''Result of `align`: `lag` (None when refused), `reason` ("ok" or why), `curve`.'''

    def __init__(self, reason, curve=None, best=None):
        self.reason = reason
        self.curve = curve or {}
        self.lag = best if reason == "ok" else None

    @property
    def ok(self):
        return self.reason == "ok"


def distance_curve(ref, other, lags, start=0):
    '''Mean grey distance of `ref` against `other[start + lag:]` for each lag (NaN where the
    window leaves `other`), and the number of non-flat reference frames used.'''
    lags = np.asarray(list(lags), dtype=np.int64)
    n = len(ref)
    rows = np.nonzero(ref.std >= FLAT_STD)[0]
    curve = np.full(len(lags), np.nan)
    if len(rows) == 0 or len(lags) == 0:
        return curve, len(rows)
    first = start + lags
    readable = (first >= 0) & (first + n <= len(other))
    if readable.any():
        idx = first[readable][:, None] + rows[None, :]
        dist = hamming(other.grey[idx], ref.grey[rows][None, :, :]) / ref.bits
        ref_index = np.broadcast_to(ref.thumb_index[rows][None, :], idx.shape)
        pixel = _pixel_rows(np.broadcast_to(ref.extreme[rows][None, :], idx.shape), ref_index,
                            other.extreme[idx], other.thumb_index[idx])
        if pixel.any():
            dist[pixel] = _pixel_as_bits(pixel_distance(ref.thumbs[ref_index[pixel]],
                                                        other.thumbs[other.thumb_index[idx][pixel]]))
        curve[readable] = dist.mean(axis=1)
    return curve, len(rows)


def align(ref, other, lags, start=0):
    '''Find the lag at which `other[start + lag:]` shows `ref`: the unique minimum of the
    distance curve, kept when it beats every lag two or more frames away by ALIGN_MARGIN.'''
    lags = sorted({int(lag) for lag in lags})
    if len(ref) < ALIGN_MIN_FRAMES:
        return Alignment("short")
    values, used = distance_curve(ref, other, lags, start)
    if used < ALIGN_MIN_FRAMES:
        return Alignment("flat")
    curve = {lag: float(v) for lag, v in zip(lags, values) if np.isfinite(v)}
    if not curve:
        return Alignment("unreadable")
    best = min(curve, key=lambda lag: (curve[lag], lag))
    if sum(1 for v in curve.values() if v == curve[best]) > 1:
        return Alignment("tie", curve, best)
    rivals = [v for lag, v in curve.items() if abs(lag - best) >= 2]
    if not rivals:
        return Alignment("no_rival", curve, best)
    return Alignment("ok" if min(rivals) - curve[best] >= ALIGN_MARGIN else "ambiguous", curve, best)
