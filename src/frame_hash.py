'''Perceptual frame hashes, distances and distance-curve alignment.

Every hash is `imagehash.phash`. Grey hashes place frames (alignment, offsets, frame
pinning, single-frame `same_picture`); a "same content" check over a window also compares
the Cb and Cr hashes of the same frames (`content_distance`). Hashes are kept packed in
uint64 words so distances over thousands of frames stay vectorised (XOR + popcount).

Distances are fractions of the hash bits (0 = identical, about 0.5 = unrelated).

pHash is unstable on a frame that is almost entirely black or almost entirely white: most of
its DCT coefficients sit at the median, so two copies of the same frame can read far apart.
`frame_distance` is the one frame-to-frame comparison: on such a frame it decides from a small
grey thumbnail instead (`pixel_distance`), stored only for frames near that case.
'''

import numpy as np
import imagehash
from PIL import Image

HASH_SIZE = 32
HASH_BITS = HASH_SIZE * HASH_SIZE

# Cb and Cr are hashed smaller: chroma holds little detail (12-frame same/other content gap:
# grey 0.228, + colour at 16 0.249, + colour at 32 0.232).
COLOUR_HASH_SIZE = 16
COLOUR_BITS = COLOUR_HASH_SIZE * COLOUR_HASH_SIZE

# Decode size for hashed frames; the hash resamples to 128x128 anyway.
FRAME_WIDTH = 160
FRAME_HEIGHT = 90

# A frame whose luma standard deviation (0-255) is below this is flat: its hash bits are
# noise, so it never takes part in an alignment.
FLAT_STD = 3.0

# Weight of the mean Cb/Cr distance in `content_distance`.
COLOUR_WEIGHT = 0.5

# An alignment is kept only when its minimum beats every lag two or more frames away by
# this fraction of the hash bits (12-frame windows: 82 % kept, 97 % exact, worst 1 frame).
ALIGN_MARGIN = 0.01

# Fewest usable (non-flat) reference frames an alignment accepts.
ALIGN_MIN_FRAMES = 6

# Same content: mean `content_distance` of a window of 6 or more frames at most this
# (same content up to 0.278, other content from 0.457).
SAME_CONTENT_MAX = 0.37

# Same picture: grey `distance` of one frame pair at most this (keeps 95 % of same frames,
# drops half of those two frames apart, other scenes from 0.436; colour separates single
# frames worse).
SAME_FRAME_MAX = 0.07

# Near-uniform frames. A luma pixel is "extreme" at or below DARK_LEVEL or at or above
# LIGHT_LEVEL; `extreme` is the fraction of such pixels, of whichever end holds more.
DARK_LEVEL = 48
LIGHT_LEVEL = 207
# From this fraction a frame is near-uniform and `frame_distance` decides by pixels. Measured on
# two pairs of identical releases (pixel difference 0.6/255): every 12+ frame run of false pHash
# differences (0.25-0.53 on identical frames) breaks into runs of at most 7 frames here, against
# 16-26 at 0.8-0.9; redrawn shots of two other pairs are all still found.
UNIFORM_FRACTION = 0.75
# Thumbnails are kept from this looser fraction, so that the other copy of a near-uniform
# frame still carries one when it reads slightly under UNIFORM_FRACTION.
THUMB_FRACTION = 0.6
THUMB_WIDTH = 32
THUMB_HEIGHT = 18
# Luma std under which the z-score stops amplifying a thumbnail's own encode noise: at 8, two
# encodes of one near-black shot (std 3-4) read 0.0045, at the SAME_PIXEL_MAX gate.
PIXEL_STD_FLOOR = 12.0
# Same picture by pixels: `pixel_distance` at most this. Identical near-uniform frames: p99.9
# 0.0012, max 0.0043; frames of redrawn dark shots mostly 0.008-0.4 (a 0.09 gate lost four of
# eight real shots).
SAME_PIXEL_MAX = 0.005


class FrameHashes:
    '''Hashes of a run of consecutive frames.

    Attributes:
        grey: (n, words) uint64, packed grey pHash bits.
        colour: (n, 2, colour words) uint64, packed Cb and Cr pHash bits, or None.
        std: (n,) float32 luma standard deviation of each frame.
        extreme: (n,) float32 fraction of near-black or near-white pixels (see DARK_LEVEL).
        thumb_index: (n,) int32 row of each frame in `thumbs`, -1 when it has none.
        thumbs: (k, THUMB_HEIGHT, THUMB_WIDTH) uint8 grey thumbnails, shared by every slice.

    Indexing with an int, a slice or an index array returns another FrameHashes.
    '''

    __slots__ = ("grey", "colour", "std", "extreme", "thumb_index", "thumbs")

    def __init__(self, grey, colour=None, std=None, extreme=None, thumb_index=None,
                 thumbs=None):
        self.grey = np.asarray(grey, dtype=np.uint64)
        self.colour = None if colour is None else np.asarray(colour, dtype=np.uint64)
        n = len(self.grey)
        self.std = (np.full(n, np.inf, dtype=np.float32) if std is None
                    else np.asarray(std, dtype=np.float32))
        self.extreme = (np.zeros(n, dtype=np.float32) if extreme is None
                        else np.asarray(extreme, dtype=np.float32))
        self.thumb_index = (np.full(n, -1, dtype=np.int32) if thumb_index is None
                            else np.asarray(thumb_index, dtype=np.int32))
        self.thumbs = (np.zeros((0, THUMB_HEIGHT, THUMB_WIDTH), np.uint8) if thumbs is None
                       else np.asarray(thumbs, dtype=np.uint8))

    def __len__(self):
        return len(self.grey)

    def __getitem__(self, key):
        if isinstance(key, (int, np.integer)):
            key = slice(int(key), int(key) + 1) if key != -1 else slice(-1, None)
        return FrameHashes(self.grey[key],
                           None if self.colour is None else self.colour[key],
                           self.std[key], self.extreme[key], self.thumb_index[key], self.thumbs)

    @property
    def bits(self):
        '''Number of hash bits per frame.'''
        return self.grey.shape[-1] * 64

    @staticmethod
    def empty(colour=False):
        '''A FrameHashes of no frame.'''
        return FrameHashes(np.zeros((0, HASH_BITS // 64), np.uint64),
                           np.zeros((0, 2, COLOUR_BITS // 64), np.uint64) if colour else None,
                           np.zeros(0, np.float32))

    @staticmethod
    def concat(parts):
        '''Join FrameHashes end to end; colour is kept only when every part has it.'''
        parts = list(parts)
        if not parts:
            return FrameHashes.empty()
        colour = (np.concatenate([p.colour for p in parts])
                  if all(p.colour is not None for p in parts) else None)
        index, thumbs, base = [], [], 0
        for p in parts:
            used = np.unique(p.thumb_index[p.thumb_index >= 0])
            remap = np.full(len(p.thumbs) + 1, -1, dtype=np.int32)
            remap[used] = np.arange(base, base + len(used), dtype=np.int32)
            index.append(remap[p.thumb_index])
            thumbs.append(p.thumbs[used])
            base += len(used)
        return FrameHashes(np.concatenate([p.grey for p in parts]), colour,
                           np.concatenate([p.std for p in parts]),
                           np.concatenate([p.extreme for p in parts]),
                           np.concatenate(index), np.concatenate(thumbs))


# -- frames ----------------------------------------------------------------------------------

def frames_from_raw(blob, width, height, channels=3):
    '''Turn raw rawvideo bytes (rgb24 or gray) into a (n, h, w[, 3]) uint8 array.

    A trailing partial frame is dropped.
    '''
    size = width * height * channels
    count = len(blob) // size
    shape = (count, height, width, channels) if channels > 1 else (count, height, width)
    return np.frombuffer(blob[:count * size], dtype=np.uint8).reshape(shape)


def grey_frames(rgb):
    '''Luma of (n, h, w, 3) RGB frames, as PIL computes it, (n, h, w) uint8.'''
    if len(rgb) == 0:
        return np.zeros((0,) + tuple(np.shape(rgb)[1:3]), dtype=np.uint8)
    return np.stack([np.asarray(Image.fromarray(np.ascontiguousarray(f), "RGB").convert("L"))
                     for f in rgb])


# -- hashing ---------------------------------------------------------------------------------

def phash(image, hash_size=HASH_SIZE):
    '''`imagehash.phash` of one PIL image (ImageHash converts it to grey).'''
    return imagehash.phash(image, hash_size=hash_size)


def pack(image_hash):
    '''Pack an ImageHash's boolean array into uint64 words.'''
    return np.packbits(np.asarray(image_hash.hash, dtype=bool).ravel()).view(np.uint64)


def phash_many(images, hash_size=HASH_SIZE):
    '''Packed grey pHash of each image.

    Args:
        images: PIL images or 2-D uint8 arrays.

    Returns:
        (n, hash_size**2 // 64) uint64 array.
    '''
    words = max(1, hash_size * hash_size // 64)
    out = [pack(phash(img if isinstance(img, Image.Image) else Image.fromarray(img), hash_size))
           for img in images]
    return np.stack(out) if out else np.zeros((0, words), np.uint64)


def hash_frames(frames, colour=False, hash_size=HASH_SIZE, colour_size=COLOUR_HASH_SIZE):
    '''Hash a run of frames.

    Args:
        frames: (n, h, w) grey or (n, h, w, 3) RGB uint8 frames.
        colour: also hash the Cb and Cr planes (RGB frames only).

    Returns:
        FrameHashes.
    '''
    frames = np.asarray(frames)
    rgb = frames.ndim == 4
    if colour and not rgb:
        raise ValueError("colour hashes need RGB frames")
    words = max(1, hash_size * hash_size // 64)
    colour_words = max(1, colour_size * colour_size // 64)
    grey, chroma, std, extreme, thumb_index, thumbs = [], [], [], [], [], []
    for frame in frames:
        img = Image.fromarray(np.ascontiguousarray(frame), "RGB" if rgb else "L")
        luma = img.convert("L") if rgb else img
        grey.append(pack(phash(luma, hash_size)))
        pixels = np.asarray(luma)
        std.append(pixels.astype(np.float32).std())
        extreme.append(extreme_fraction(pixels))
        if extreme[-1] >= THUMB_FRACTION:
            thumb_index.append(len(thumbs))
            thumbs.append(thumbnail(luma))
        else:
            thumb_index.append(-1)
        if colour:
            _, cb, cr = img.convert("YCbCr").split()
            chroma.append([pack(phash(cb, colour_size)), pack(phash(cr, colour_size))])
    n = len(grey)
    return FrameHashes(np.stack(grey) if n else np.zeros((0, words), np.uint64),
                       (np.asarray(chroma, np.uint64).reshape(n, 2, colour_words) if colour else None),
                       np.asarray(std, np.float32), np.asarray(extreme, np.float32),
                       np.asarray(thumb_index, np.int32),
                       np.stack(thumbs) if thumbs else None)


def extreme_fraction(pixels):
    '''Fraction of a grey frame's pixels at the near-black end or the near-white end, whichever
    holds more.'''
    pixels = np.asarray(pixels)
    if pixels.size == 0:
        return 0.0
    dark = np.count_nonzero(pixels <= DARK_LEVEL)
    light = np.count_nonzero(pixels >= LIGHT_LEVEL)
    return max(dark, light) / pixels.size


def thumbnail(luma):
    '''THUMB_WIDTH x THUMB_HEIGHT area-averaged grey thumbnail of a PIL "L" image, uint8.'''
    return np.asarray(luma.resize((THUMB_WIDTH, THUMB_HEIGHT), Image.BOX), dtype=np.uint8)


# -- distances -------------------------------------------------------------------------------

def hamming(a, b):
    '''Differing bits between packed hashes (words on the last axis); broadcasts.

    FrameHashes arguments compare their grey hashes.
    '''
    a = a.grey if isinstance(a, FrameHashes) else a
    b = b.grey if isinstance(b, FrameHashes) else b
    return np.bitwise_count(np.bitwise_xor(a, b)).sum(axis=-1, dtype=np.int64)


def distance(a, b):
    '''Per-frame raw grey pHash distance of two equal-length FrameHashes, as a fraction of the
    bits. Comparisons go through `frame_distance`.'''
    return hamming(a.grey, b.grey) / a.bits


def pixel_distance(a, b):
    '''(1 - Pearson correlation) / 2 of grey thumbnails (..., h, w), each z-scored by its own
    mean and std (std floored at PIXEL_STD_FLOOR): 0 same, about 0.5 unrelated, 1 inverted.'''
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    axes = (-2, -1)
    za = (a - a.mean(axis=axes, keepdims=True)) / np.maximum(a.std(axis=axes, keepdims=True),
                                                             PIXEL_STD_FLOOR)
    zb = (b - b.mean(axis=axes, keepdims=True)) / np.maximum(b.std(axis=axes, keepdims=True),
                                                             PIXEL_STD_FLOOR)
    return np.square(za - zb).mean(axis=axes) / 4.0


def _pixel_rows(a_extreme, a_index, b_extreme, b_index):
    '''Pairs decided by pixels: either side near-uniform, both carrying a thumbnail.'''
    return (((a_extreme >= UNIFORM_FRACTION) | (b_extreme >= UNIFORM_FRACTION))
            & (a_index >= 0) & (b_index >= 0))


def _pixel_as_bits(pixel):
    '''A `pixel_distance` on the pHash bit-fraction scale, piecewise linear: 0 stays 0,
    SAME_PIXEL_MAX lands on SAME_FRAME_MAX, 0.5 (unrelated) on 0.5, capped there.'''
    pixel = np.asarray(pixel, dtype=np.float64)
    low = pixel * (SAME_FRAME_MAX / SAME_PIXEL_MAX)
    high = SAME_FRAME_MAX + (pixel - SAME_PIXEL_MAX) * ((0.5 - SAME_FRAME_MAX)
                                                        / (0.5 - SAME_PIXEL_MAX))
    return np.minimum(np.where(pixel <= SAME_PIXEL_MAX, low, high), 0.5)


def frame_distance(a, b):
    '''Per-frame distance of two equal-length FrameHashes on the grey pHash scale.

    The grey pHash fraction of differing bits, except where either frame is near-uniform
    (`extreme` >= UNIFORM_FRACTION) and both carry a thumbnail: there `pixel_distance`
    decides, rescaled by `_pixel_as_bits` so every pHash threshold keeps its meaning.
    '''
    dist = hamming(a.grey, b.grey) / a.bits
    rows = _pixel_rows(a.extreme, a.thumb_index, b.extreme, b.thumb_index)
    if rows.any():
        dist = np.asarray(dist, dtype=np.float64)
        pixel = pixel_distance(a.thumbs[a.thumb_index[rows]], b.thumbs[b.thumb_index[rows]])
        dist[rows] = _pixel_as_bits(pixel)
    return dist


def same_picture(a, b):
    '''Per-frame verdicts (bool array) that two equal-length FrameHashes show the same picture.'''
    return frame_distance(a, b) <= SAME_FRAME_MAX


def content_distance(a, b):
    '''Per-frame content distance of two equal-length FrameHashes, as a fraction of the bits.

    (grey + COLOUR_WEIGHT x mean(Cb, Cr)) / (1 + COLOUR_WEIGHT), each part as a fraction of
    its own bits; `frame_distance` alone where it decides by pixels. Both sides need colour.
    '''
    if a.colour is None or b.colour is None:
        raise ValueError("content_distance needs colour hashes on both sides")
    grey = hamming(a.grey, b.grey) / a.bits
    chroma = hamming(a.colour, b.colour).mean(axis=-1) / (a.colour.shape[-1] * 64)
    content = (grey + COLOUR_WEIGHT * chroma) / (1.0 + COLOUR_WEIGHT)
    rows = _pixel_rows(a.extreme, a.thumb_index, b.extreme, b.thumb_index)
    if rows.any():
        # chroma hashes of a near-uniform frame are as unstable as its grey one
        content = np.asarray(content, dtype=np.float64)
        content[rows] = frame_distance(a[np.nonzero(rows)[0]], b[np.nonzero(rows)[0]])
    return content


def window_content_distance(ref, other, offset=0):
    '''Mean `content_distance` of `ref` against `other[offset : offset + len(ref)]`.

    Flat reference frames are left out. NaN when the window leaves `other` or no frame
    is usable.
    '''
    if offset < 0 or offset + len(ref) > len(other):
        return float("nan")
    rows = np.nonzero(ref.std >= FLAT_STD)[0]
    if len(rows) == 0:
        return float("nan")
    return float(np.mean(content_distance(ref[rows], other[rows + offset])))


# -- alignment -------------------------------------------------------------------------------

class Alignment:
    '''Result of `align`.

    Attributes:
        lag: the chosen lag, or None when refused.
        reason: "ok", or why it was refused: "short", "flat", "unreadable", "tie",
            "no_rival", "ambiguous".
        curve: {lag: mean grey distance} over the readable lags.
        best: the lag of the minimum (also set when refused, when there is one).
        distance: the minimum mean distance.
        margin: second-best distance two or more lags away minus the minimum.
        rival: the lag of that second best.
        frames: reference frames used (non-flat).
        start: the `start` the lags are counted from.
    '''

    def __init__(self, reason, curve=None, best=None, margin=None, rival=None, frames=0,
                 start=0):
        self.reason = reason
        self.start = start
        self.curve = curve or {}
        self.best = best
        self.lag = best if reason == "ok" else None
        self.distance = None if best is None else self.curve[best]
        self.margin = margin
        self.rival = rival
        self.frames = frames

    @property
    def ok(self):
        return self.reason == "ok"

    def __repr__(self):
        margin = "None" if self.margin is None else f"{self.margin:.4f}"
        return (f"Alignment(lag={self.lag} reason={self.reason} best={self.best} "
                f"margin={margin} rival={self.rival} frames={self.frames})")


def distance_curve(ref, other, lags, start=0):
    '''Mean `frame_distance` of `ref` against `other[start + lag :]`, for each lag.

    Flat reference frames (std < FLAT_STD) are left out, the same rows for every lag.

    Returns:
        (curve, used): curve is a float array aligned with `lags`, NaN where the window
        leaves `other`; used is the number of reference frames compared.
    '''
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


def align(ref, other, lags, start=0, margin=ALIGN_MARGIN, min_frames=ALIGN_MIN_FRAMES):
    '''Find the lag at which `other` shows the frames of `ref`.

    Lag `l` compares ref[i] with other[start + l + i]. The minimum of the mean-distance
    curve is kept only when it is unique and beats every lag two or more frames away by
    `margin` (fraction of the bits); adjacent lags are not gated, the argmin decides.

    Returns:
        Alignment.
    '''
    lags = sorted({int(lag) for lag in lags})
    if len(ref) < min_frames:
        return Alignment("short", frames=len(ref), start=start)
    values, used = distance_curve(ref, other, lags, start)
    if used < min_frames:
        return Alignment("flat", frames=used, start=start)
    curve = {lag: float(v) for lag, v in zip(lags, values) if np.isfinite(v)}
    if not curve:
        return Alignment("unreadable", frames=used, start=start)
    best = min(curve, key=lambda lag: (curve[lag], lag))
    low = curve[best]
    if sum(1 for v in curve.values() if v == low) > 1:
        return Alignment("tie", curve, best, frames=used, start=start)
    rivals = {lag: v for lag, v in curve.items() if abs(lag - best) >= 2}
    if not rivals:
        return Alignment("no_rival", curve, best, frames=used, start=start)
    rival = min(rivals, key=lambda lag: (rivals[lag], lag))
    gap = rivals[rival] - low
    return Alignment("ok" if gap >= margin else "ambiguous", curve, best, gap, rival, used,
                     start)
