'''Frame alignment of two videos: the scene changes of each file, matched picture to picture by pHash.'''

from decimal import Decimal
from fractions import Fraction

import numpy as np
from scenedetect import ContentDetector, FrameTimecode

import frame_compare

FRAMES_AROUND_CUT = 3
SAME_PICTURE_MAX_DISTANCE = 10

def find_scene_changes(video_obj, position, search_seconds):
    frame_duration = frame_compare.get_frame_duration(video_obj)
    """ Example:
        position Decimal("600.5"), search_seconds Decimal("10") on a 24000/1001 video
        -> [14158, [37, 140, 251, 402], [<pHash of frame 14158>, <pHash of frame 14159>, ...]]
        list_cut holds indexes in list_hash: frame 14158+37 is the first frame of a new scene
    """
    first_frame = max(0, round((position-search_seconds)/frame_duration))
    frames, list_hash = frame_compare.decode_frames(video_obj, first_frame, round(2*search_seconds/frame_duration))
    frame_rate = Fraction(video_obj.video["FrameRate"])
    detector = ContentDetector(min_scene_len=1)
    list_cut = []
    for index, frame in enumerate(frames):
        list_cut.extend(timecode.frame_num for timecode in detector.process_frame(FrameTimecode(index, fps=frame_rate), np.ascontiguousarray(frame[..., ::-1])))
    return [first_frame, list_cut, list_hash]

def match_scene_changes(video_obj_a, video_obj_b, position_a, position_b, search_seconds):
    '''Return [list_match, list_unmatched]: the scene changes of A around position_a and of B around position_b (seconds), paired when their frames are the same pictures.'''
    """ Example:
        position_a Decimal("600.5"), position_b Decimal("601.501"), search_seconds Decimal("10")
        -> [[[14195, 14219], [14298, 14322]], [[14240, None], [None, 14341]]]
        a pair is [frame of A, frame of B], the first frame of the same new scene in each file;
        a scene change without match keeps None on the side of the other file
    """
    first_frame_a, list_cut_a, list_hash_a = find_scene_changes(video_obj_a, position_a, search_seconds)
    first_frame_b, list_cut_b, list_hash_b = find_scene_changes(video_obj_b, position_b, search_seconds)
    list_pair = []
    for cut_a in list_cut_a:
        list_same = []
        for cut_b in list_cut_b:
            distance = frame_compare.get_distance(list_hash_a, cut_a-FRAMES_AROUND_CUT, list_hash_b, cut_b-FRAMES_AROUND_CUT, 2*FRAMES_AROUND_CUT)
            if distance is not None and distance <= SAME_PICTURE_MAX_DISTANCE:
                list_same.append(cut_b)
        if len(list_same) == 1:
            list_pair.append([cut_a, list_same[0]])
    list_cut_b_paired = [cut_b for cut_a, cut_b in list_pair]
    list_pair = [[cut_a, cut_b] for cut_a, cut_b in list_pair if list_cut_b_paired.count(cut_b) == 1]
    list_match = [[first_frame_a+cut_a, first_frame_b+cut_b] for cut_a, cut_b in list_pair]
    list_unmatched = [[first_frame_a+cut_a, None] for cut_a in list_cut_a if cut_a not in [pair[0] for pair in list_pair]]
    list_unmatched.extend([None, first_frame_b+cut_b] for cut_b in list_cut_b if cut_b not in [pair[1] for pair in list_pair])
    return [list_match, list_unmatched]

def get_scene_change_delays(video_obj_a, video_obj_b, list_match):
    '''Return [[delay_ms, frame_a, frame_b], ...]: for each matched scene change, its time in B minus its time in A, in milliseconds.'''
    frame_duration_ms_a = frame_compare.get_frame_duration(video_obj_a)*Decimal(1000)
    frame_duration_ms_b = frame_compare.get_frame_duration(video_obj_b)*Decimal(1000)
    return [[Decimal(frame_b)*frame_duration_ms_b-Decimal(frame_a)*frame_duration_ms_a, frame_a, frame_b] for frame_a, frame_b in list_match]
