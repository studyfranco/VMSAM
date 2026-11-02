'''Image comparison toolbox: frames decoded small on their exact grid, black borders cropped, pHash and Hamming distance.'''

from decimal import Decimal

import numpy as np
import imagehash
from PIL import Image

import tools

WIDTH = 160
HEIGHT = 90
BLACK_LEVEL = 16
HASH_SIZE = 8

def get_frame_duration(video_obj):
    '''Return the duration of one frame of a CFR video in seconds, as a Decimal.'''
    if video_obj.video.get("FrameRate_Num") and video_obj.video.get("FrameRate_Den"):
        frame_duration = Decimal(video_obj.video["FrameRate_Den"])/Decimal(video_obj.video["FrameRate_Num"])
    else:
        frame_duration = Decimal(1)/Decimal(video_obj.video["FrameRate"])
    return frame_duration

def decode_frames(video_obj, first_frame, number_frames):
    '''Return [frames, list_hash]: frames first_frame to first_frame+number_frames-1, cropped to the picture, and the pHash of each.'''
    seek_time = max(Decimal(0), (Decimal(first_frame)-Decimal("0.5"))*get_frame_duration(video_obj))
    cmd_decode = [tools.software["ffmpeg"], "-hide_banner", "-nostdin", "-threads", "3", "-ss", f"{seek_time:.6f}", "-i", video_obj.filePath,
                  "-map", f"0:{video_obj.video['StreamOrder']}", "-frames:v", str(number_frames), "-fps_mode", "passthrough",
                  "-vf", f"scale={WIDTH}:{HEIGHT}:flags=area", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
    stdout, stderror, exitCode = tools.launch_cmdExt_with_timeout_reload(cmd_decode, 2, 600)
    number_decoded = len(stdout)//(WIDTH*HEIGHT*3)
    frames = np.frombuffer(stdout[:number_decoded*WIDTH*HEIGHT*3], dtype=np.uint8).reshape(number_decoded, HEIGHT, WIDTH, 3)
    if number_decoded:
        peak = frames.max(axis=(0, 3))
        list_row = np.flatnonzero(peak.max(axis=1) > BLACK_LEVEL)
        list_column = np.flatnonzero(peak.max(axis=0) > BLACK_LEVEL)
        if len(list_row) and len(list_column):
            frames = frames[:, list_row[0]:list_row[-1]+1, list_column[0]:list_column[-1]+1]
    list_hash = [imagehash.phash(Image.fromarray(frame), hash_size=HASH_SIZE) for frame in frames]
    return [frames, list_hash]

def get_distance(list_hash_a, start_a, list_hash_b, start_b, number_frames):
    '''Return the worst Hamming distance between number_frames pHash of A from start_a and of B from start_b, None when a side has not enough frames.'''
    distance = None
    if 0 <= start_a and start_a+number_frames <= len(list_hash_a) and 0 <= start_b and start_b+number_frames <= len(list_hash_b):
        distance = max(list_hash_a[start_a+position]-list_hash_b[start_b+position] for position in range(number_frames))
    return distance
