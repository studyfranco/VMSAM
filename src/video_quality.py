"""Rank two renditions of the same episode without a pristine source.

`get_best_quality_video` (video.py) used to decide only from a two-way VMAF
scan (each rendition scored once as the other's reference, then averaged).
That is an asymmetric full-reference comparison with no clean source on
either side, and two defects made it worse: the result was returned as the
string "1"/"2" while both call sites test `== 1` (an int), so the VMAF
verdict never actually reached the caller, and an empty/unparsed VMAF sample
raised `statistics.StatisticsError` out of a function the caller cannot
afford to have raise.

This module tries two cheaper, more explainable signals first and keeps the
two-way VMAF scan only as the last-resort fallback:

1. `heuristic_pick` -- encode-parameter heuristics from metadata already on
   the two `Video` objects: pixel count, codec generation, bits per pixel.
   No decode at all; decisive on the common, easy cases (1080p vs 720p,
   Blu-ray vs web, a two-tier-older codec).
2. `no_reference_pick` -- a bounded, no-reference scan with ffmpeg's
   `blurdetect`/`blockdetect` filters, run once per rendition (never one as
   the other's reference) over the sample windows already chosen for the
   delay check, at `-threads 2`.
3. The caller's own two-way VMAF scan, used only when both of the above are
   inconclusive or fail to measure.

See VMSAM_HELP_AI/architect/drafts/BEST_VIDEO_METHOD.md for the sources and
the alternatives this rejected.
"""
import re
from statistics import mean

# Rough codec-generation ranking. An unknown codec is treated as mid-tier
# (never penalized for simply being unrecognized).
CODEC_RANK = {
    "av1": 4,
    "vp9": 3, "hevc": 3, "h265": 3,
    "avc": 2, "h264": 2, "vp8": 2,
    "mpeg-4 visual": 1, "mpeg4": 1,
    "mpeg video": 0, "mpeg-2 video": 0,
}

RESOLUTION_DECISIVE_RATIO = 1.3
CODEC_TIER_GAP_DECISIVE = 2
BITS_PER_PIXEL_DECISIVE_RATIO = 1.25
NO_REFERENCE_MARGIN = 0.05


def _codec_rank(video_obj):
    """Return the codec-generation rank of a Video object, mid-tier if unknown."""
    fmt = str(video_obj.video.get("Format", "")).strip().lower()
    return CODEC_RANK.get(fmt, 1)


def _bits_per_pixel(video_obj, get_bitrate_fn):
    """Return bitrate per pixel-second, or None when it cannot be computed."""
    try:
        bitrate = float(get_bitrate_fn(video_obj.video))
    except Exception:
        return None
    scale = video_obj.get_scale()
    fps = video_obj.get_fps()
    if not scale or not fps or scale[0] <= 0 or scale[1] <= 0 or fps <= 0:
        return None
    pixels_per_second = scale[0] * scale[1] * fps
    if pixels_per_second <= 0:
        return None
    return bitrate / pixels_per_second


def heuristic_pick(video_obj_1, video_obj_2, get_bitrate_fn):
    """Decide from encode parameters alone. Returns (winner, reason) or (None, reason)."""
    scale_1 = video_obj_1.get_scale()
    scale_2 = video_obj_2.get_scale()
    pixels_1 = scale_1[0] * scale_1[1] if scale_1 else None
    pixels_2 = scale_2[0] * scale_2[1] if scale_2 else None
    codec_1 = _codec_rank(video_obj_1)
    codec_2 = _codec_rank(video_obj_2)

    if pixels_1 and pixels_2 and pixels_1 != pixels_2:
        bigger = 1 if pixels_1 > pixels_2 else 2
        ratio = max(pixels_1, pixels_2) / min(pixels_1, pixels_2)
        bigger_codec = codec_1 if bigger == 1 else codec_2
        smaller_codec = codec_2 if bigger == 1 else codec_1
        if ratio >= RESOLUTION_DECISIVE_RATIO and bigger_codec >= smaller_codec:
            return bigger, f"resolution {ratio:.2f}x, equal-or-better codec on the bigger side"

    if codec_1 != codec_2 and abs(codec_1 - codec_2) >= CODEC_TIER_GAP_DECISIVE:
        better = 1 if codec_1 > codec_2 else 2
        pixels_better = pixels_1 if better == 1 else pixels_2
        pixels_worse = pixels_2 if better == 1 else pixels_1
        if not (pixels_better and pixels_worse and pixels_worse > pixels_better * RESOLUTION_DECISIVE_RATIO):
            return better, f"codec generation gap ({codec_1} vs {codec_2})"

    if pixels_1 and pixels_2 and pixels_1 == pixels_2 and codec_1 == codec_2:
        bpp_1 = _bits_per_pixel(video_obj_1, get_bitrate_fn)
        bpp_2 = _bits_per_pixel(video_obj_2, get_bitrate_fn)
        if bpp_1 and bpp_2:
            hi = 1 if bpp_1 > bpp_2 else 2
            if max(bpp_1, bpp_2) >= min(bpp_1, bpp_2) * BITS_PER_PIXEL_DECISIVE_RATIO:
                return hi, f"bits/pixel {bpp_1:.4f}/{bpp_2:.4f} at matched resolution and codec"

    return None, "no decisive encode-parameter signal"


def _common_filters(video_obj_1, video_obj_2):
    """Return ffmpeg -vf fragments reconciling fps/scale, as the VMAF/PSNR scans do."""
    framerate_1 = video_obj_1.get_fps()
    framerate_2 = video_obj_2.get_fps()
    scale_1 = video_obj_1.get_scale()
    scale_2 = video_obj_2.get_scale()
    filters = []
    if framerate_1 and framerate_2 and framerate_1 != framerate_2:
        filters.append(f"fps=fps={min(framerate_1, framerate_2)}")
    if scale_1 and scale_2 and (scale_1[0] != scale_2[0] or scale_1[1] != scale_2[1]):
        bigger = scale_1 if scale_1[0] * scale_1[1] > scale_2[0] * scale_2[1] else scale_2
        filters.append(f"scale={bigger[0]}:{bigger[1]}")
    return filters


def _scan_no_reference(video_obj, begin, time_by_test, filters, tools_module):
    """Run one bounded, single-pass blurdetect/blockdetect scan. Returns (blur, block) or None."""
    cmd = [tools_module.software["ffmpeg"], "-ss", begin, "-t", time_by_test,
           "-i", video_obj.filePath, "-map", f"0:{video_obj.video['StreamOrder']}"]
    vf = ",".join(filters + ["blurdetect", "blockdetect"])
    cmd += ["-vf", vf, "-threads", "2", "-f", "null", "-"]
    _, stderr, _ = tools_module.launch_cmdExt(cmd)
    text = stderr.decode("utf-8", "replace")
    blur = re.search(r'blur mean:\s*([\d.]+)', text)
    block = re.search(r'block mean:\s*([\d.]+)', text)
    if not blur or not block:
        return None
    return float(blur.group(1)), float(block.group(1))


def no_reference_pick(video_obj_1, video_obj_2, begins_video, time_by_test, tools_module):
    """Bounded no-reference scan, each rendition decoded on its own. Returns (winner, reason) or (None, reason)."""
    if not begins_video:
        return None, "no sample window to scan"
    filters = _common_filters(video_obj_1, video_obj_2)
    blur_1_vals, block_1_vals, blur_2_vals, block_2_vals = [], [], [], []
    for begin_1, begin_2 in begins_video:
        result_1 = _scan_no_reference(video_obj_1, begin_1, time_by_test, filters, tools_module)
        result_2 = _scan_no_reference(video_obj_2, begin_2, time_by_test, filters, tools_module)
        if result_1 is None or result_2 is None:
            return None, "no-reference scan produced no parseable output"
        blur_1_vals.append(result_1[0]); block_1_vals.append(result_1[1])
        blur_2_vals.append(result_2[0]); block_2_vals.append(result_2[1])

    blur_1, block_1 = mean(blur_1_vals), mean(block_1_vals)
    blur_2, block_2 = mean(blur_2_vals), mean(block_2_vals)

    blur_margin = abs(blur_1 - blur_2) / max(blur_1, blur_2, 1e-9)
    block_margin = abs(block_1 - block_2) / max(block_1, block_2, 1e-9)
    # Lower blur mean is sharper; higher block mean is less blocky. Both measured
    # empirically on local ffmpeg 9.0.2 (crf10 vs crf45 of the same source: blur
    # 5.04 vs 7.35, block 97.1 vs 84.4).
    blur_winner = 1 if blur_1 < blur_2 else 2
    block_winner = 1 if block_1 > block_2 else 2
    summary = f"blur {blur_1:.3f}/{blur_2:.3f}, block {block_1:.3f}/{block_2:.3f}"

    if blur_margin < NO_REFERENCE_MARGIN and block_margin < NO_REFERENCE_MARGIN:
        return None, f"no-reference scan inconclusive ({summary})"
    if blur_winner != block_winner and blur_margin >= NO_REFERENCE_MARGIN and block_margin >= NO_REFERENCE_MARGIN:
        return None, f"no-reference scan disagreement, blur favors {blur_winner}, block favors {block_winner} ({summary})"

    winner = blur_winner if blur_margin >= block_margin else block_winner
    return winner, f"no-reference scan {summary}"


def pick_best_video(video_obj_1, video_obj_2, begins_video, time_by_test, get_bitrate_fn, tools_module, vmaf_fallback):
    """Return 1 or 2, the better rendition, logging the rule that decided.

    Never raises to the caller except re-raising KeyboardInterrupt/SystemExit;
    any other measurement failure falls back to `vmaf_fallback()`.
    """
    try:
        winner, reason = heuristic_pick(video_obj_1, video_obj_2, get_bitrate_fn)
        if winner is not None:
            tools_module.log_always(f"best_video: winner={winner} rule=encode-heuristic ({reason})\n")
            return winner

        winner, reason = no_reference_pick(video_obj_1, video_obj_2, begins_video, time_by_test, tools_module)
        if winner is not None:
            tools_module.log_always(f"best_video: winner={winner} rule=no-reference-scan ({reason})\n")
            return winner

        tools_module.log_always(
            f"best_video: encode-heuristic and no-reference scan inconclusive ({reason}); "
            f"falling back to two-way VMAF\n")
        return vmaf_fallback()
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as exc:
        tools_module.log_always(
            f"best_video: measurement failed ({type(exc).__name__}: {exc}); "
            f"falling back to two-way VMAF\n")
        return vmaf_fallback()
