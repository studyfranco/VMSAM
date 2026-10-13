"""Locate a break between two videos inside a time window by comparing frames.

Each frame is decoded as a 32x32 grey picture and reduced to a 64-bit DCT
perceptual hash (pHash). The reference frames are matched to the target frames
inside a band around the diagonal; the target zone that collects the largest
mismatch cost is taken as the break. When no frame can be hashed, ffmpeg scene
detection on the target gives a fallback zone.

Importer: `get_cut_time.py` (`FrameComparer(...).find_scene_gap_requirements()`).
"""

import re
from sys import stderr

import numpy as np
from scipy.fft import dct

import tools

HASH_SIDE = 32       # decoded frame side, in pixels
DCT_SIDE = 8         # low-frequency block kept from the DCT
NO_MATCH = 1_000_000  # larger than any 64-bit Hamming distance
SHOWINFO_TIME = re.compile(r"pts_time:([0-9]+\.[0-9]+)")


def phash64(frame):
    """Return the 64-bit perceptual hash of one grey frame.

    Takes a (HASH_SIDE, HASH_SIDE) array of grey levels. The 2-D DCT is computed,
    its top-left 8x8 block kept, and bit k (row-major) is set when coefficient k
    is above the median of the 63 non-DC coefficients. Returns an int.
    """
    coefficients = dct(dct(frame, norm="ortho", axis=0), norm="ortho", axis=1)
    low = coefficients[:DCT_SIDE, :DCT_SIDE].flatten()
    # The DC term carries only the mean brightness: it is left out of the median.
    median = np.median(low[1:])
    return sum(1 << bit for bit, value in enumerate(low) if value > median)


def phash64_frames(raw_bytes, side=HASH_SIDE):
    """Return the pHash of every complete frame in a raw grey buffer.

    Takes bytes holding consecutive side x side 8-bit frames (ffmpeg `rawvideo`,
    `gray`); a trailing partial frame is ignored. Returns a list of ints.
    """
    frame_size = side * side
    frame_count = len(raw_bytes) // frame_size
    frames = np.frombuffer(raw_bytes[:frame_count * frame_size], dtype=np.uint8)
    frames = frames.reshape((frame_count, side, side)).astype(np.float32)
    return [phash64(frame) for frame in frames]


def hamming64(hash_a, hash_b):
    """Return the number of differing bits between two 64-bit hashes."""
    return (hash_a ^ hash_b).bit_count()


def band_match_costs(ref_hashes, tgt_hashes, band_width):
    """Match each reference frame to its closest target frame inside a band.

    Reference frame i is compared with target frames i-band_width..i+band_width;
    the closest one (first on a tie) receives the distance as cost. Returns one
    accumulated cost per target frame: a target frame that only bad matches
    land on is a frame the reference does not carry.
    """
    tgt_count = len(tgt_hashes)
    costs = [0] * tgt_count
    for ref_index, ref_hash in enumerate(ref_hashes):
        best_index, best_distance = None, NO_MATCH
        first = max(0, ref_index - band_width)
        last = min(tgt_count - 1, ref_index + band_width)
        for tgt_index in range(first, last + 1):
            distance = hamming64(ref_hash, tgt_hashes[tgt_index])
            if distance < best_distance:
                best_index, best_distance = tgt_index, distance
        if best_index is not None:
            costs[best_index] += best_distance
    return costs


def trailing_sums(values, window):
    """Return, for each index j, the sum of values[j-window+1..j] (fewer at the start)."""
    sums = [0] * len(values)
    running = 0
    for index, value in enumerate(values):
        running += value
        if index >= window:
            running -= values[index - window]
        sums[index] = running
    return sums


def align_and_find_gap(ref_hashes, tgt_hashes, band_width, max_search_frames, fps):
    """Return the target frame zone (start, end) that matches the reference worst.

    Takes the two hash lists (only the first max_search_frames of each are used),
    the band half-width in frames and the integer sampling rate. The costs of
    `band_match_costs` are summed over about a third of a second and the zone is
    centred on the highest sum. Returns indices into the target list, or None
    when either list is empty.
    """
    ref = ref_hashes[:max_search_frames]
    tgt = tgt_hashes[:max_search_frames]
    if not ref or not tgt:
        return None
    window = max(3, fps // 3)
    smoothed = trailing_sums(band_match_costs(ref, tgt, band_width), window)
    center = max(range(len(tgt)), key=lambda index: smoothed[index])
    half = max(2, window // 2)
    return max(0, center - half), min(len(tgt) - 1, center + half)


def scene_times_from_showinfo(showinfo_text, offset_sec):
    """Return the times of the frames listed by ffmpeg `showinfo`, shifted by offset_sec."""
    return [float(match.group(1)) + offset_sec for match in SHOWINFO_TIME.finditer(showinfo_text)]


def zone_around_median(times, start_sec, end_sec):
    """Return a (start, end) zone around the median scene change, or None if there is none.

    The half-width is a fifth of the window, kept within 0.2..2 s, and the zone
    is clipped to [start_sec, end_sec].
    """
    if not times:
        return None
    center = times[len(times) // 2]
    half = max(0.2, min(2.0, max(0.5, end_sec - start_sec) * 0.2))
    return max(start_sec, center - half), min(end_sec, center + half)


class FrameComparer:
    """Compare two videos frame by frame in a time window to locate a break.

    ref_path, tgt_path: the reference and target files.
    start_sec, end_sec: the window, in seconds, the same on both files.
    fps: sampling rate of the decoded frames (an integer, at least 1).
    band_width: how far, in frames, a reference frame may match off the diagonal.
    max_search_frames: frames compared on each side (at least 8).
    scene_threshold: ffmpeg scene score above which the fallback sees a change.
    """

    def __init__(self, ref_path, tgt_path, start_sec, end_sec,
                 fps=10, band_width=20, max_search_frames=50, debug=False,
                 scene_threshold=0.30):
        self.ref_path = ref_path
        self.tgt_path = tgt_path
        self.start_sec = float(start_sec)
        self.end_sec = float(end_sec)
        self.fps = int(max(1, fps))
        self.band_width = int(max(1, band_width))
        self.max_search_frames = int(max(8, max_search_frames))
        self.debug = debug
        self.scene_threshold = float(scene_threshold)

    def read_grey_frames(self, path, start_sec, dur_sec):
        """Decode [start_sec, start_sec+dur_sec] of path as raw 32x32 grey frames at self.fps.

        Returns the bytes ffmpeg wrote; on an ffmpeg error the partial output is
        still returned, since a short window is usable even when truncated.
        """
        cmd = [
            tools.software["ffmpeg"], "-v", "error", "-nostdin",
            "-ss", f"{start_sec}", "-t", f"{dur_sec}", "-i", path,
            "-vf", f"scale={HASH_SIDE}:{HASH_SIDE},fps={self.fps},format=gray",
            "-f", "rawvideo", "-pix_fmt", "gray", "pipe:1"
        ]
        stdout, _, exit_code = tools.launch_cmdExt_no_test(cmd)
        if exit_code != 0 and self.debug:
            stderr.write(f"[frame_compare] ffmpeg returned {exit_code}, partial data used\n")
        return stdout

    def scene_gap_fallback(self, start_sec, end_sec):
        """Return a (start, end) zone in seconds around the target's median scene change, or None.

        Runs ffmpeg scene detection on the target over [start_sec, end_sec].
        """
        dur = max(0.5, end_sec - start_sec)
        cmd = [
            tools.software["ffmpeg"], "-hide_banner", "-nostdin",
            "-ss", f"{start_sec}", "-t", f"{dur}", "-i", self.tgt_path,
            "-vf", f"select='gt(scene,{self.scene_threshold})',showinfo",
            "-f", "null", "-"
        ]
        # showinfo writes on stderr; its times are relative to the -ss seek.
        _, showinfo, _ = tools.launch_cmdExt_no_test(cmd)
        times = scene_times_from_showinfo(showinfo.decode("utf-8", errors="ignore"), start_sec)
        return zone_around_median(times, start_sec, end_sec)

    def find_scene_gap_requirements(self, before_common=2, after_common=3):
        """Locate the break in the window and return it on the target's timeline.

        The window is padded by up to 4 s on each side, both files are hashed,
        and the worst-matching zone is taken (scene detection when nothing could
        be hashed). Returns {"start_frame", "end_frame", "start_time",
        "end_time"} (frames at self.fps, times in seconds), or None when neither
        method finds a zone. before_common and after_common are accepted for
        the caller's signature and unused.
        """
        dur = max(0.5, self.end_sec - self.start_sec)
        pad = min(4.0, dur / 2.0)
        start = max(0.0, self.start_sec - pad)
        padded_dur = dur + 2 * pad

        ref_hashes = phash64_frames(self.read_grey_frames(self.ref_path, start, padded_dur))
        tgt_hashes = phash64_frames(self.read_grey_frames(self.tgt_path, start, padded_dur))
        if self.debug:
            stderr.write(f"[frame_compare] hashed frames: ref={len(ref_hashes)} tgt={len(tgt_hashes)}\n")

        gap = align_and_find_gap(ref_hashes, tgt_hashes, self.band_width,
                                 self.max_search_frames, self.fps)
        if gap is None:
            zone = self.scene_gap_fallback(start, start + padded_dur)
            if zone is None:
                return None
            # Snap the fallback times onto the sampling grid, like the frame path.
            gap = tuple(int(round((seconds - start) * self.fps)) for seconds in zone)
        elif self.debug:
            stderr.write(f"[frame_compare] gap target frames: {gap[0]}..{gap[1]}\n")

        start_time = start + gap[0] / float(self.fps)
        end_time = start + gap[1] / float(self.fps)
        return {
            "start_frame": int(round(start_time * self.fps)),
            "end_frame": int(round(end_time * self.fps)),
            "start_time": start_time,
            "end_time": end_time
        }
