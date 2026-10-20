'''Perceptual frame hashes: decode, hash, and the hash container.

Every hash is `imagehash.phash` -- this is the ONLY module that computes it. Grey hashes place
frames (alignment, offsets, frame pinning, single-frame "same picture"); a "same content" check
over a window also compares the Cb and Cr hashes of the same frames. Hashes are kept packed in
uint64 words so distances over thousands of frames stay vectorised (XOR + popcount) by the
comparison side.

Comparisons, distances and alignment (on top of the hashes this module produces) live in
`frame_compare.py`, not here: this module never imports it (no cycle), and nothing in it compares
two hashes to each other.
'''

import numpy as np
import imagehash
from PIL import Image

HASH_SIZE = 32
HASH_BITS = HASH_SIZE * HASH_SIZE

# Cb and Cr are hashed smaller: chroma holds little detail, so a small hash adds
# almost as much separation between same and other content as a full-size one.
COLOUR_HASH_SIZE = 16
COLOUR_BITS = COLOUR_HASH_SIZE * COLOUR_HASH_SIZE

# Decode size for hashed frames; the hash resamples to 128x128 anyway.
FRAME_WIDTH = 160
FRAME_HEIGHT = 90

# Near-uniform frames. A luma pixel is "extreme" at or below DARK_LEVEL or at or above
# LIGHT_LEVEL; `extreme` is the fraction of such pixels, of whichever end holds more. Stored per
# hashed frame so the comparison side can decide, per pair, whether to trust pHash or pixels.
DARK_LEVEL = 48
LIGHT_LEVEL = 207
# Thumbnails are kept from this fraction up, so the comparison side's own (looser or tighter)
# near-uniform gate always has one available when it needs it.
THUMB_FRACTION = 0.6
THUMB_WIDTH = 32
THUMB_HEIGHT = 18


class frame_hashes:
    '''Hashes of a run of consecutive frames.

    Attributes:
        grey: (n, words) uint64, packed grey pHash bits.
        colour: (n, 2, colour words) uint64, packed Cb and Cr pHash bits, or None.
        std: (n,) float32 luma standard deviation of each frame.
        extreme: (n,) float32 fraction of near-black or near-white pixels (see DARK_LEVEL).
        thumb_index: (n,) int32 row of each frame in `thumbs`, -1 when it has none.
        thumbs: (k, THUMB_HEIGHT, THUMB_WIDTH) uint8 grey thumbnails, shared by every slice.

    Indexing with an int, a slice or an index array returns another frame_hashes.
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
        return frame_hashes(self.grey[key],
                           None if self.colour is None else self.colour[key],
                           self.std[key], self.extreme[key], self.thumb_index[key], self.thumbs)

    @property
    def bits(self):
        '''Number of hash bits per frame.'''
        return self.grey.shape[-1] * 64

    @staticmethod
    def empty(colour=False):
        '''A frame_hashes of no frame.'''
        return frame_hashes(np.zeros((0, HASH_BITS // 64), np.uint64),
                           np.zeros((0, 2, COLOUR_BITS // 64), np.uint64) if colour else None,
                           np.zeros(0, np.float32))

    @staticmethod
    def concat(parts):
        '''Join frame_hashes end to end; colour is kept only when every part has it.'''
        parts = list(parts)
        if not parts:
            return frame_hashes.empty()
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
        return frame_hashes(np.concatenate([p.grey for p in parts]), colour,
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


def hash_frames(frames, colour=False, hash_size=HASH_SIZE, colour_size=COLOUR_HASH_SIZE):
    '''Hash a run of frames.

    Args:
        frames: (n, h, w) grey or (n, h, w, 3) RGB uint8 frames.
        colour: also hash the Cb and Cr planes (RGB frames only).

    Returns:
        frame_hashes.
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
    return frame_hashes(np.stack(grey) if n else np.zeros((0, words), np.uint64),
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
