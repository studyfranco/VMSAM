'''Common anchors of two videos: the scene changes both show, matched frame to frame by pHash.'''

import re

import numpy as np
import imagehash
from PIL import Image
from scenedetect import ContentDetector, FrameTimecode

import tools

SCENE_SEARCH_SECONDS = 10
FRAMES_AROUND_SCENE_CHANGE = 12
SHIFT_SEARCH_FRAMES = 12
CHUNK_SECONDS = 60
MAX_ADJUSTMENT_FRAMES = 2
MIN_SCENE_CHANGES = 3
SAME_PICTURE_MAX_DISTANCE = 10
GROUP_CUT_THRESHOLD = 27.0
WIDTH = 160
HEIGHT = 80
PTS_TIME = re.compile(r"pts_time:\s*(-?[0-9.]+)")

class FrameAlignmentError(Exception):
    pass

def decode_group(video_obj, start_s, n_frames):
    cmd_decode = [tools.software["ffmpeg"], "-hide_banner", "-nostdin", "-threads", "3", "-copyts",
                  "-ss", f"{max(0.0, start_s):.6f}", "-i", video_obj.filePath,
                  "-map", f"0:{video_obj.video['StreamOrder']}", "-frames:v", str(n_frames), "-an", "-sn", "-dn",
                  "-vf", f"showinfo,crop=iw/2:iw/4,scale={WIDTH}:{HEIGHT}:flags=area,format=rgb24",
                  "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
    stdout, stderror, exitCode = tools.launch_cmdExt_with_timeout_reload(cmd_decode, 2, 600)
    frame_size = WIDTH*HEIGHT*3
    times = [float(pts_time) for pts_time in PTS_TIME.findall(stderror.decode("utf-8", errors="ignore"))]
    count = min(len(stdout)//frame_size, len(times))
    frames = np.frombuffer(stdout[:count*frame_size], dtype=np.uint8).reshape(count, HEIGHT, WIDTH, 3)
    return frames, np.array(times[:count])

def group_cuts(frames):
    detector = ContentDetector(threshold=GROUP_CUT_THRESHOLD, min_scene_len=1)
    cuts = []
    for index, frame in enumerate(frames):
        for timecode in detector.process_frame(FrameTimecode(index, fps=24.0), np.ascontiguousarray(frame[..., ::-1])):
            cuts.append(int(timecode.frame_num))
    return cuts

def find_common_anchors(master_obj, candidate_obj, master_spans, nominal_shift_frames, max_shift_frames):
    '''Return [list_anchor, majority_shift, vote_count]: the scene changes both videos show in master_spans, matched by pHash.'''
    list_anchor = []
    """ Example:
        master_spans [[1490.0, 1510.0]] in seconds on the master clock, nominal_shift_frames 24, max_shift_frames 12
        -> [[[35836, 35860, 24, 2], [35932, 35956, 24, 0], [36007, 36031, 24, 4]], 24, 3]
        anchor = [master frame, candidate frame, shift, Hamming distance summed over the 2*FRAMES_AROUND_SCENE_CHANGE frames]
        the candidate shows at master frame + shift what the master shows at master frame
        majority_shift is None under MIN_SCENE_CHANGES anchors or without a strict majority
    """
    same_frame_rate = master_obj.video.get("FrameRate_Mode") == "CFR" and candidate_obj.video.get("FrameRate_Mode") == "CFR" and master_obj.video.get("FrameRate") == candidate_obj.video.get("FrameRate")
    if same_frame_rate:
        frame_duration = 1.0/float(master_obj.video["FrameRate"])
        margin_frame = max_shift_frames+FRAMES_AROUND_SCENE_CHANGE
        for span_start, span_end in master_spans:
            chunk_start = max(0.0, span_start)
            while chunk_start < span_end:
                chunk_end = min(span_end, chunk_start+CHUNK_SECONDS)
                number_frame_chunk = int((chunk_end-chunk_start)/frame_duration)
                frames_master, times_master = decode_group(master_obj, chunk_start-FRAMES_AROUND_SCENE_CHANGE*frame_duration, number_frame_chunk+2*FRAMES_AROUND_SCENE_CHANGE)
                frames_candidate, times_candidate = decode_group(candidate_obj, chunk_start+(nominal_shift_frames-margin_frame)*frame_duration, number_frame_chunk+2*margin_frame)
                hashes_master = [imagehash.phash(Image.fromarray(frame)) for frame in frames_master]
                hashes_candidate = [imagehash.phash(Image.fromarray(frame)) for frame in frames_candidate]
                cuts_candidate = group_cuts(frames_candidate)
                for cut_master in group_cuts(frames_master):
                    index_candidate = int(np.argmin(np.abs(times_candidate-(times_master[cut_master]+nominal_shift_frames*frame_duration)))) if len(times_candidate) else 0
                    in_chunk = chunk_start <= times_master[cut_master] < chunk_end
                    in_both = any(abs(cut_candidate-index_candidate) <= max_shift_frames for cut_candidate in cuts_candidate)
                    in_range = cut_master >= FRAMES_AROUND_SCENE_CHANGE and cut_master+FRAMES_AROUND_SCENE_CHANGE <= len(hashes_master) and index_candidate >= margin_frame and index_candidate+margin_frame <= len(hashes_candidate)
                    if in_chunk and in_both and in_range:
                        list_distance = []
                        for shift in range(-max_shift_frames, max_shift_frames+1):
                            list_distance.append(sum(hashes_master[cut_master+position]-hashes_candidate[index_candidate+shift+position] for position in range(-FRAMES_AROUND_SCENE_CHANGE, FRAMES_AROUND_SCENE_CHANGE)))
                        best_distance = min(list_distance)
                        if list_distance.count(best_distance) == 1 and best_distance <= SAME_PICTURE_MAX_DISTANCE*2*FRAMES_AROUND_SCENE_CHANGE:
                            best_shift = list_distance.index(best_distance)-max_shift_frames
                            master_frame = round(times_master[cut_master]/frame_duration)
                            candidate_frame = round(times_candidate[index_candidate+best_shift]/frame_duration)
                            list_anchor.append([master_frame, candidate_frame, nominal_shift_frames+best_shift, int(best_distance)])
                chunk_start = chunk_end

    list_shift = [anchor[2] for anchor in list_anchor]
    majority_shift = None
    vote_count = 0
    if list_shift:
        most_common_shift = max(set(list_shift), key=list_shift.count)
        if len(list_shift) >= MIN_SCENE_CHANGES and list_shift.count(most_common_shift)*2 > len(list_shift):
            majority_shift = most_common_shift
            vote_count = list_shift.count(most_common_shift)
    votes_per_shift = {shift: list_shift.count(shift) for shift in sorted(set(list_shift))}
    tools.log_always(f"[frame_align] {candidate_obj.filePath} on {master_obj.filePath}: same frame rate {same_frame_rate}, nominal shift {nominal_shift_frames} frame(s), votes per shift {votes_per_shift}, majority shift {majority_shift}\n")
    return [list_anchor, majority_shift, vote_count]
