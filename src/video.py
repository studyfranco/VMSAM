'''
Created on 23 Apr 2022

@author: francois
'''

from os import path,remove
import os # For os.path.exists, os.remove in detect_upscale
from sys import stderr
import sys # For sys.stderr in detect_upscale
from threading import RLock
from time import strftime,gmtime,sleep
import tools
import re
import json
# from decimal import Decimal # Not strictly needed for detect_upscale, but often used in this file.

ffmpeg_pool_audio_convert = None
ffmpeg_pool_big_job = None
path_to_livmaf_model = "" #Nothing if it use the default
number_cut = 5
percent_time_by_test_video_quality_from_cut = 25

# --- Upscale Detection Function ---
def detect_upscale(video_obj, segment_begin, segment_duration):
    """
    Detects if a video segment is likely upscaled by comparing it with a
    downscaled-then-upscaled version using SSIM.
    """
    if not video_obj.video or 'Width' not in video_obj.video or 'Height' not in video_obj.video:
        sys.stderr.write("DETECT_UPSCALE: Video object missing dimension attributes.\n")
        return False, (video_obj.video.get('Height') if video_obj.video else 0)

    try:
        W_orig = int(video_obj.video['Width'])
        H_orig = int(video_obj.video['Height'])
        if W_orig <=0 or H_orig <=0: 
            sys.stderr.write(f"DETECT_UPSCALE: Invalid original dimensions W_orig={W_orig}, H_orig={H_orig}.\n")
            return False, H_orig
    except ValueError:
        sys.stderr.write("DETECT_UPSCALE: Could not parse original dimensions.\n")
        return False, (video_obj.video.get('Height') if video_obj.video else 0)

    test_heights = [720, 540, 480, 360]
    detected_native_height = H_orig
    is_likely_upscaled = False
    SSIM_THRESHOLD = 0.985
    ffmpeg_path = tools.software["ffmpeg"]
    
    video_stream_map = f"0:{video_obj.video['StreamOrder']}"
    
    safe_segment_begin = re.sub(r'[^0-9a-zA-Z]', '_', str(segment_begin))
    safe_segment_duration = re.sub(r'[^0-9a-zA-Z]', '_', str(segment_duration))
    
    all_temp_files_for_function = []

    for h_test in test_heights:
        if H_orig <= h_test:
            continue

        w_test = (int(W_orig * h_test / H_orig) // 2) * 2
        if w_test <= 0:
            sys.stderr.write(f"DETECT_UPSCALE: Calculated w_test={w_test} is invalid for h_test={h_test}.\n")
            continue

        base_name_for_temp = f"{video_obj.fileBaseName}_upscaletest_s{safe_segment_begin}_d{safe_segment_duration}_h{h_test}"
        orig_segment_path = path.join(tools.tmpFolder, f"{base_name_for_temp}_orig.mkv")
        rescaled_segment_path = path.join(tools.tmpFolder, f"{base_name_for_temp}_rescaled.mkv")
        ssim_log_filename = f"{base_name_for_temp}_ssim.log" 
        ssim_log_path_full = path.join(tools.tmpFolder, ssim_log_filename) 
        
        current_iter_temp_files = [orig_segment_path, rescaled_segment_path, ssim_log_path_full]
        all_temp_files_for_function.extend(current_iter_temp_files)

        try:
            cmd_orig = [
                ffmpeg_path, '-y', '-nostdin',
                '-i', video_obj.filePath,
                '-ss', str(segment_begin), 
                '-t', str(segment_duration),
                '-map', video_stream_map,
                '-c', 'copy', 
                '-threads', '1', 
                '-an', '-sn', 
                orig_segment_path
            ]
            tools.launch_cmdExt(cmd_orig)

            vf_filter = f"scale={w_test}:{h_test},scale={W_orig}:{H_orig}:flags=bicubic"
            cmd_rescale = [
                ffmpeg_path, '-y', '-nostdin',
                '-i', orig_segment_path, 
                '-vf', vf_filter,
                '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '18', 
                '-threads', '1',
                '-an', '-sn', 
                rescaled_segment_path
            ]
            tools.launch_cmdExt(cmd_rescale)

            cmd_ssim = [
                ffmpeg_path, '-threads', '1', '-nostdin',
                '-i', orig_segment_path,
                '-i', rescaled_segment_path,
                '-lavfi', f"[0:v][1:v]ssim=stats_file={ssim_log_filename}", 
                '-f', 'null', '-'
            ]
            _stdout_ssim, stderr_ssim, _exit_code_ssim = tools.launch_cmdExt(cmd_ssim, cwd=tools.tmpFolder)

            ssim_all_score = None
            if os.path.exists(ssim_log_path_full):
                with open(ssim_log_path_full, 'r') as f_log:
                    log_content = f_log.read()
                match = re.search(r"All:([\d\.]+)", log_content) 
                if match:
                    ssim_all_score = float(match.group(1))
            
            if ssim_all_score is None and stderr_ssim: 
                stderr_content = stderr_ssim.decode("utf-8", errors='ignore')
                match_stderr = re.search(r"SSIM Y:.*? All:([\d\.]+)", stderr_content)
                if match_stderr:
                    ssim_all_score = float(match_stderr.group(1))

            if ssim_all_score is not None:
                sys.stderr.write(f"DETECT_UPSCALE: Test H_orig={H_orig}, h_test={h_test}, SSIM: {ssim_all_score}\n")
                if ssim_all_score >= SSIM_THRESHOLD:
                    is_likely_upscaled = True
                    detected_native_height = h_test
                    break 
            else:
                sys.stderr.write(f"DETECT_UPSCALE: Failed to parse SSIM score for h_test={h_test}.\n")

        except Exception as e:
            sys.stderr.write(f"DETECT_UPSCALE: Error during upscale detection for {video_obj.filePath} at {h_test}p: {e}\n")
        finally:
            for f_path in current_iter_temp_files:
                if os.path.exists(f_path):
                    try: os.remove(f_path)
                    except OSError as e_rm: sys.stderr.write(f"DETECT_UPSCALE: Error removing temp file {f_path}: {e_rm}\n")
    
    for f_path_final in all_temp_files_for_function: 
         if os.path.exists(f_path_final):
            try: os.remove(f_path_final)
            except OSError: pass

    return is_likely_upscaled, detected_native_height
# --- End Upscale Detection Function ---

class video():
    '''
    classdocs
    '''


    def __init__(self, fileFolder, fileName):
        '''
        Constructor
        '''
        self.fileFolder = fileFolder
        self.fileName = fileName
        self.fileBaseName = path.splitext(fileName)[0]
        self.filePath = path.join(fileFolder, fileName)
        if (not tools.file_exists(self.filePath)):
            raise Exception(self.filePath+" not exist")
        self.mediadata = None
        self.mkvmergedata = None
        self.audios = None
        self.audiodesc = None
        self.commentary = None
        self.video = None
        self.subtitles = None
        self.video_quality = None
        self.tmpFiles = {}
        self.ffmpeg_progress_audio = []
        self.delays = {}
        self.lastCutAsDefault = False
        self.delayFirstMethodAbort = {}
        self.shiftCuts = None
        self.sameAudioMD5UseForCalculation = []
    
    def get_mediadata(self):
        stdout, stderror, exitCode = tools.launch_cmdExt([tools.software["mediainfo"], "--Output=JSON", self.filePath])
        if exitCode != 0:
            raise Exception("Error with {} during the mediadata: {}".format(self.filePath,stderror.decode("UTF-8")))
        self.mediadata = json.loads(stdout.decode("UTF-8"))
        stdout, stderror, exitCode = tools.launch_cmdExt([tools.software["mkvmerge"],"-i", "-F", "json", self.filePath])
        if exitCode != 0:
            raise Exception("Error with {} during the mkvmerge metadata: {}".format(self.filePath,stderror.decode("UTF-8")))
        self.mkvmergedata = json.loads(stdout.decode("UTF-8"))
        properties_track = {}
        for track in self.mkvmergedata['tracks']:
            if str(track['id']) in properties_track:
                raise Exception(f"{self.filePath} have tracks with the same ids")
            properties_track[str(track['id'])] = track['properties']
        ffprobe_data = tools.extract_ffmpeg_type_dict_all(self.filePath)
        self.audios = {}
        self.subtitles = {}
        self.commentary = {}
        self.audiodesc = {}
        for data in self.mediadata['media']['track']:
            data['MD5'] = ''
            data["keep"] = True
            if 'StreamOrder' in data:
                try:
                    data['properties'] = properties_track[data['StreamOrder']]
                    data['ffprobe'] = ffprobe_data[int(data['StreamOrder'])]
                except:
                    raise Exception(f"{self.filePath} have problematic track id")
            if data['@type'] == 'Video':
                if self.video != None:
                    raise Exception(f"Multiple video in the same file {self.filePath}, I can't compare and merge they")
                else:
                    self.video = data
            elif data['@type'] == 'Audio': 
                if 'Language' in data:
                    language = data['Language'].split("-")[0]
                else:
                    language = "und"
                if ('Title' in data and 'commentary' in data['Title'].lower()) or ("flag_commentary" in data['properties'] and data['properties']["flag_commentary"]):
                    if language in self.commentary:
                        self.commentary[language].append(data)
                    else:
                        self.commentary[language] = [data]
                elif ('Title' in data and re.match(r".* *\[{0,1}audio {0,1}description\]{0,1} *.*", data["Title"].lower()) ) or ("flag_visual_impaired" in data['properties'] and data['properties']["flag_visual_impaired"]):
                    if language in self.audiodesc:
                        self.audiodesc[language].append(data)
                    else:
                        self.audiodesc[language] = [data]
                else:
                    data["compatible"] = True
                    if language in self.audios:
                        self.audios[language].append(data)
                    else:
                        self.audios[language] = [data]
            elif data['@type'] == 'Text':
                if 'Language' in data:
                    language = data['Language'].split("-")[0]
                    if language in self.subtitles:
                        self.subtitles[language].append(data)
                    else:
                        self.subtitles[language] = [data]
        if len(self.audios) == 0:
            raise Exception(f"No audio usable to compare the file {self.filePath}")
        if "und" in self.audios and tools.default_language_for_undetermine not in self.audios:
            # This step is linked to mergeVideo.generate_merge_command_insert_ID_audio_track_to_remove_and_new_und_language
            self.audios[tools.default_language_for_undetermine] = self.audios["und"]
            
    def get_best_video(self,data_video_1,data_video_2):
        stderr.write("!"*40+"\n")
        stderr.write(f'Multiple video in the same file {self.filePath}, I will compare the {data_video_1["StreamOrder"]} et {data_video_2["StreamOrder"]} track\n')
        stderr.write("If your video are not sync, the result can be random.\n")
        stderr.write("!"*40+"\n")
        
        
            
    def get_fps(self):
        if 'FrameRate' in self.video:
            return float(self.video['FrameRate'])
        else:
            return None
    
    def get_scale(self):
        if 'Width' in self.video and 'Height' in self.video:
            return [int(self.video['Width']), int(self.video['Height'])]
        else:
            return None
        
    def get_video_duration(self):
        if 'FrameCount' in self.video:
            return float((float(self.video['FrameCount'])/float(self.video['FrameRate'])))
        else:
            return None
    
    def extract_audio_in_part(self,language,exportParam,cutTime=None,asDefault=False):
        if (not self.lastCutAsDefault) or (not asDefault):
            self.lastCutAsDefault = asDefault
            global ffmpeg_pool_audio_convert
            self.wait_end_ffmpeg_progress_audio()
            nameFilesExtract = []
            if 'audio' in self.tmpFiles:
                self.remove_tmp_files(type_file="audio")
            self.tmpFiles['audio'] = nameFilesExtract
    
            baseCommand = [tools.software["ffmpeg"], "-y", "-threads", str(tools.core_to_use), "-nostdin", "-i", self.filePath, "-vn"]
            if exportParam['Format'] == 'WAV':
                if 'codec' in exportParam:
                    baseCommand.extend(["-c:a", exportParam['codec']])
            elif 'codec' in exportParam:
                baseCommand.extend(["-acodec", exportParam['codec']])
            else:
                baseCommand.extend(["-acodec", exportParam['Format'].lower().replace('-','')])
            if 'BitRate' in exportParam:
                baseCommand.extend(["-ab", exportParam['BitRate']])
            if 'SamplingRate' in exportParam:
                baseCommand.extend(["-ar", exportParam['SamplingRate']])
            if 'Channels' in exportParam:
                baseCommand.extend(["-ac", exportParam['Channels']])
            audio_pos_file = 0
            wait_end_big_job()
            if cutTime == None:
                for audio in self.audios[language]:
                    if audio["compatible"]:
                        nameFilesExtractCut = []
                        nameFilesExtract.append(nameFilesExtractCut)
                        audio["audio_pos_file"] = audio_pos_file
                        audio_pos_file += 1
                        nameOutFile = path.join(tools.tmpFolder,self.fileBaseName+"."+str(audio['StreamOrder'])+".1"+"."+exportParam['Format'].lower().replace('-',''))
                        nameFilesExtractCut.append(nameOutFile)
                        cmd = baseCommand.copy()
                        cmd.extend(["-map", "0:"+str(audio['StreamOrder']), nameOutFile])
                        self.ffmpeg_progress_audio.append(ffmpeg_pool_audio_convert.apply_async(tools.launch_cmdExt, (cmd,)))
            else:
                for audio in self.audios[language]:
                    if audio["compatible"]:
                        nameFilesExtractCut = []
                        nameFilesExtract.append(nameFilesExtractCut)
                        audio["audio_pos_file"] = audio_pos_file
                        audio_pos_file += 1
                        cutNumber = 0
                        for cut in cutTime:
                            nameOutFile = path.join(tools.tmpFolder,self.fileBaseName+"."+str(audio['StreamOrder'])+"."+str(cutNumber)+"."+exportParam['Format'].lower().replace('-',''))
                            nameFilesExtractCut.append(nameOutFile)
                            cmd = baseCommand.copy()
                            cmd.extend(["-map", "0:"+str(audio['StreamOrder']), "-ss", cut[0], "-t", cut[1] , nameOutFile])
                            self.ffmpeg_progress_audio.append(ffmpeg_pool_audio_convert.apply_async(tools.launch_cmdExt, (cmd,)))
                            cutNumber += 1
            
    def remove_tmp_files(self,type_file=None):
        if type == None:
            for key,list_tmp in self.tmpFiles:
                for files in list_tmp:
                    for file in files:
                        remove(file)
            self.tmpFiles = {}
        else:
            for files in self.tmpFiles[type_file]:
                for file in files:
                    remove(file)
            self.tmpFiles[type_file] = []
                
    def wait_end_ffmpeg_progress_audio(self):
        while len(self.ffmpeg_progress_audio) > 0:
            ffmpeg_job = self.ffmpeg_progress_audio.pop(0)
            ffmpeg_job.get()
        self.ffmpeg_progress_audio = []
        
    def calculate_md5_streams(self):
        if self.mediadata == None:
            self.get_mediadata()
        task_audio = {}
        for language, data in self.audios.items():
            task_audio[language] = []
            for audio in data:
                task_audio[language].append(ffmpeg_pool_audio_convert.apply_async(md5_calculator,(self.filePath,audio["StreamOrder"])))
                
        task_commentary = {}
        for language, data in self.commentary.items():
            task_commentary[language] = []
            for audio in data:
                task_commentary[language].append(ffmpeg_pool_audio_convert.apply_async(md5_calculator,(self.filePath,audio["StreamOrder"])))

        task_audio_desc = {}
        for language, data in self.audiodesc.items():
            task_audio_desc[language] = []
            for audio in data:
                task_audio_desc[language].append(ffmpeg_pool_audio_convert.apply_async(md5_calculator,(self.filePath,audio["StreamOrder"])))

        task_subtitle = {}
        for language, data in self.subtitles.items():
            task_subtitle[language] = []
            for subtitle in data:
                task_subtitle[language].append(ffmpeg_pool_audio_convert.apply_async(md5_calculator,(self.filePath,subtitle["StreamOrder"])))
        
        for language, data in task_audio.items():
            i=0
            for audio in data:
                result = audio.get()
                if result[1] != None:
                    self.audios[language][i]['MD5'] = result[1]
                else:
                    stderr.write(f"Error with {self.filePath} during the md5 calculation of the stream {result[0]}")
                i += 1

        for language, data in task_commentary.items():
            i=0
            for audio in data:
                result = audio.get()
                if result[1] != None:
                    self.commentary[language][i]['MD5'] = result[1]
                else:
                    stderr.write(f"Error with {self.filePath} during the md5 calculation of the stream {result[0]}")
                i += 1
                
        for language, data in task_audio_desc.items():
            i=0
            for audio in data:
                result = audio.get()
                if result[1] != None:
                    self.audiodesc[language][i]['MD5'] = result[1]
                else:
                    stderr.write(f"Error with {self.filePath} during the md5 calculation of the stream {result[0]}")
                i += 1

        for language, data in task_subtitle.items():
            i=0
            for subtitle in data:
                result = subtitle.get()
                if result[1] != None:
                    self.subtitles[language][i]['MD5'] = result[1]
                else:
                    stderr.write(f"Error with {self.filePath} during the md5 calculation of the stream {result[0]}")
                i += 1

    def calculate_md5_streams_split(self):
        if self.mediadata == None:
            self.get_mediadata()
        
        length_video = float(self.video['Duration'])
        if length_video > 20:
            length_video = length_video-10.0
        task_audio = {}
        for language, data in self.audios.items():
            task_audio[language] = []
            for audio in data:
                task_audio[language].append(ffmpeg_pool_audio_convert.apply_async(md5_calculator,(self.filePath,audio["StreamOrder"],10,length_video,float(audio['Duration']))))
                
        task_commentary = {}
        for language, data in self.commentary.items():
            task_commentary[language] = []
            for audio in data:
                task_commentary[language].append(ffmpeg_pool_audio_convert.apply_async(md5_calculator,(self.filePath,audio["StreamOrder"],10,length_video,float(audio['Duration']))))

        task_audio_desc = {}
        for language, data in self.audiodesc.items():
            task_audio_desc[language] = []
            for audio in data:
                task_audio_desc[language].append(ffmpeg_pool_audio_convert.apply_async(md5_calculator,(self.filePath,audio["StreamOrder"],10,length_video,float(audio['Duration']))))

        dic_index_data_sub_codec = tools.extract_ffmpeg_type_dict(self.filePath)
        task_subtitle = {}
        for language, data in self.subtitles.items():
            task_subtitle[language] = []
            for subtitle in data:
                if dic_index_data_sub_codec[int(subtitle["StreamOrder"])]["codec_name"] != None:
                    codec = dic_index_data_sub_codec[int(subtitle["StreamOrder"])]["codec_name"].lower()
                    if codec in tools.sub_type_not_encodable:
                        task_subtitle[language].append(ffmpeg_pool_audio_convert.apply_async(md5_calculator,(self.filePath,subtitle["StreamOrder"],10,length_video,float(subtitle['Duration']))))
                    else:
                        task_subtitle[language].append(ffmpeg_pool_audio_convert.apply_async(subtitle_text_md5,(self.filePath,subtitle["StreamOrder"])))
                else:
                    task_subtitle[language].append(ffmpeg_pool_audio_convert.apply_async(md5_calculator,(self.filePath,subtitle["StreamOrder"],10,length_video,float(subtitle['Duration']))))
        
        for language, data in task_audio.items():
            i=0
            for audio in data:
                result = audio.get()
                if result[1] != None:
                    self.audios[language][i]['MD5'] = result[1]
                else:
                    stderr.write(f"Error with {self.filePath} during the md5 calculation of the stream {result[0]}")
                i += 1

        for language, data in task_commentary.items():
            i=0
            for audio in data:
                result = audio.get()
                if result[1] != None:
                    self.commentary[language][i]['MD5'] = result[1]
                else:
                    stderr.write(f"Error with {self.filePath} during the md5 calculation of the stream {result[0]}")
                i += 1
                
        for language, data in task_audio_desc.items():
            i=0
            for audio in data:
                result = audio.get()
                if result[1] != None:
                    self.audiodesc[language][i]['MD5'] = result[1]
                else:
                    stderr.write(f"Error with {self.filePath} during the md5 calculation of the stream {result[0]}")
                i += 1

        for language, data in task_subtitle.items():
            i=0
            for subtitle in data:
                result = subtitle.get()
                if result[1] != None:
                    self.subtitles[language][i]['MD5'] = result[1]
                else:
                    stderr.write(f"Error with {self.filePath} during the md5 calculation of the stream {result[0]}")
                i += 1

"""
Preparation function
"""
def generate_begin_and_length_by_segment(min_video_duration_in_sec):
    if min_video_duration_in_sec > 540:
        begin_in_second = 120
        length_time = int((min_video_duration_in_sec-240)/number_cut)
    elif min_video_duration_in_sec > 60:
        begin_in_second = 30
        length_time = int((min_video_duration_in_sec-45)/number_cut)
    elif min_video_duration_in_sec > 5:
        begin_in_second = 0
        length_time = int(min_video_duration_in_sec-2/number_cut)

    return float(begin_in_second),length_time

def generate_cut_with_begin_length(begin_in_second,length_time,length_time_converted):
    list_cut_begin_length = []
    for i in range(0,number_cut):
        time_second_begin, time_milisecond_begin = str(begin_in_second).split(".")
        list_cut_begin_length.append([strftime('%H:%M:%S',gmtime(int(time_second_begin)))+"."+time_milisecond_begin,length_time_converted])
        begin_in_second += length_time
    
    return list_cut_begin_length
    
def generate_cut_to_compare_video_quality(begin_in_second_video_1,begin_in_second_video_2,length_time):
    begins_video_for_compare_quality = []
    for i in range(0,number_cut):
        time_second_begin_video_1, time_milisecond_begin_video_1 = str(begin_in_second_video_1).split(".")
        time_second_begin_video_2, time_milisecond_begin_video_2 = str(begin_in_second_video_2).split(".")
        begins_video_for_compare_quality.append([strftime('%H:%M:%S',gmtime(int(time_second_begin_video_1)))+"."+time_milisecond_begin_video_1,strftime('%H:%M:%S',gmtime(int(time_second_begin_video_2)))+"."+time_milisecond_begin_video_2])
        begin_in_second_video_1 += length_time
        begin_in_second_video_2 += length_time
        
    return begins_video_for_compare_quality

def generate_time_compare_video_quality(length_time):
    if percent_time_by_test_video_quality_from_cut >= 100:
        time_by_test_best_quality = length_time
    else:
        time_by_test_best_quality = int(length_time*percent_time_by_test_video_quality_from_cut/100)+1
    
    return time_by_test_best_quality

def get_begin_time_with_millisecond(delay,beginInSecBeforeDelay):
    begining_in_second = int(beginInSecBeforeDelay)
    delayUseNegative = delay < 0
    delayUseSec = int(abs(delay)/1000)
    delayUseMillisecond = abs(delay)%1000
    if delayUseNegative:
        if delayUseSec > beginInSecBeforeDelay or (delayUseSec == beginInSecBeforeDelay and delayUseMillisecond > 0):
            raise Exception("Need to be update for negative delay and negative Time")
        else:
            begining_in_second -= delayUseSec
            if delayUseMillisecond > 0:
                begining_in_second -= 1
                delayUseMillisecond = 1000-delayUseMillisecond
                if delayUseMillisecond < 10:
                    begining_in_millisecond = "."+"0"*2
                elif delayUseMillisecond < 100:
                    begining_in_millisecond = "."+"0"*1
                else:
                    begining_in_millisecond = "."
                begining_in_millisecond += str(delayUseMillisecond)
            else:
                begining_in_millisecond = ""
    else:
        begining_in_second += delayUseSec
        if delayUseMillisecond > 0:
            if delayUseMillisecond < 10:
                begining_in_millisecond = "."+"0"*2
            elif delayUseMillisecond < 100:
                begining_in_millisecond = "."+"0"*1
            else:
                begining_in_millisecond = "."
            begining_in_millisecond += str(delayUseMillisecond)
        else:
            begining_in_millisecond = ""
    return begining_in_second, begining_in_millisecond

big_job_in_porgress = RLock()
def wait_end_big_job():
    global big_job_in_porgress
    #while big_job_in_porgress.locked():
    big_job_in_porgress.acquire()
    big_job_in_porgress.release()

def big_job_waiter():
    global ffmpeg_pool_audio_convert
    for i in range(0,tools.core_to_use-1):
        ffmpeg_pool_audio_convert.apply_async(sleep, (30,))
    ffmpeg_pool_audio_convert.apply_async(sleep, (0.00000000001,)).get()

def get_best_quality_video(video_obj_1, video_obj_2, begins_video, time_by_test):
    import re
    from statistics import mean

    # Upscale detection first
    # begins_video is a list of pairs: [['HH:MM:SS.mmm', 'HH:MM:SS.mmm'], ...]
    # segment_begin_v1/v2 should be strings. time_by_test is also a string 'HH:MM:SS' or float/int.
    # detect_upscale expects segment_begin and segment_duration as strings.
    segment_begin_v1_str = str(begins_video[0][0])
    segment_begin_v2_str = str(begins_video[0][1])
    time_by_test_str = str(time_by_test) # Ensure time_by_test is string for detect_upscale
    
    is_upscaled_1, native_height_1 = detect_upscale(video_obj_1, segment_begin_v1_str, time_by_test_str)
    is_upscaled_2, native_height_2 = detect_upscale(video_obj_2, segment_begin_v2_str, time_by_test_str)

    if is_upscaled_1 and not is_upscaled_2:
        return "2"  # video_obj_2 is preferred (not upscaled)
    elif not is_upscaled_1 and is_upscaled_2:
        return "1"  # video_obj_1 is preferred (not upscaled)
    elif is_upscaled_1 and is_upscaled_2:
        if native_height_1 > native_height_2:
            return "1"  # video_obj_1 preferred (upscaled from higher res)
        elif native_height_2 > native_height_1:
            return "2"  # video_obj_2 preferred (upscaled from higher res)
        else:
            # Both upscaled from same native height, proceed to VMAF
            sys.stderr.write(f"DETECT_UPSCALE: Both videos appear upscaled from same native height {native_height_1} or original. Proceeding to VMAF.\n")
            pass
    else:
        # Neither detected as upscaled, proceed to VMAF
        sys.stderr.write(f"DETECT_UPSCALE: Neither video detected as upscaled (or detection failed). Proceeding to VMAF.\n")
        pass

    # VMAF comparison logic (existing code)
    # Ensure time_by_test is suitable for ffmpeg -t (can be seconds or HH:MM:SS)
    # The original code uses strftime for time_by_test_best_quality_converted, which is passed as time_by_test here.
    # So, time_by_test should be 'HH:MM:SS' string.
    ffmpeg_VMAF_1_vs_2 = [tools.software["ffmpeg"], 
                           "-ss", segment_begin_v1_str, "-t", time_by_test_str, "-i", video_obj_1.filePath, 
                           "-ss", segment_begin_v2_str, "-t", time_by_test_str, "-i", video_obj_2.filePath,
                           "-lavfi", "[0:{}][1:{}]libvmaf=n_threads={}:log_fmt=json".format(video_obj_1.video['StreamOrder'],video_obj_2.video['StreamOrder'],tools.core_to_use)+path_to_livmaf_model,
                           "-threads", str(tools.core_to_use), "-f", "null","-map", f"0:{video_obj_1.video['StreamOrder']}", "-map", f"1:{video_obj_2.video['StreamOrder']}", "-"]
    
    framerate_video_obj_1 = video_obj_1.get_fps()
    framerate_video_obj_2 = video_obj_2.get_fps()
    scale_video_obj_1 = video_obj_1.get_scale()
    scale_video_obj_2 = video_obj_2.get_scale()
    filter_modifications = []
    if framerate_video_obj_1 != None and framerate_video_obj_2 != None and framerate_video_obj_1 != framerate_video_obj_2:
        if framerate_video_obj_1 > framerate_video_obj_2:
            filter_modifications.append(f'fps=fps={framerate_video_obj_2}')
        else:
            filter_modifications.append(f'fps=fps={framerate_video_obj_1}')
    if scale_video_obj_1 != None and scale_video_obj_2 != None and (scale_video_obj_1[0] != scale_video_obj_2[0] or scale_video_obj_1[1] != scale_video_obj_2[1]):
        if (scale_video_obj_1[0]*scale_video_obj_1[1]) > (scale_video_obj_2[0]*scale_video_obj_2[1]):
            filter_modifications.append(f'scale={scale_video_obj_1[0]}:{scale_video_obj_1[1]}')
        else:
            filter_modifications.append(f'scale={scale_video_obj_2[0]}:{scale_video_obj_2[1]}')
    if len(filter_modifications):
        ffmpeg_VMAF_1_vs_2[13] = "-filter_complex"
        ffmpeg_VMAF_1_vs_2[14] = "[0:{}]{}[0];[1:{}]{}[1]; [0][1]libvmaf=n_threads={}:log_fmt=json".format(video_obj_1.video['StreamOrder'],", ".join(filter_modifications),video_obj_2.video['StreamOrder'],", ".join(filter_modifications),tools.core_to_use)+path_to_livmaf_model

    ffmpeg_VMAF_2_vs_1 = ffmpeg_VMAF_1_vs_2.copy()
    ffmpeg_VMAF_2_vs_1[6] = video_obj_2.filePath
    ffmpeg_VMAF_2_vs_1[12] = video_obj_1.filePath
    out_1_vs_2 = []
    out_2_vs_1 = []
    values_1_vs_2 = []
    values_2_vs_1 = []
    begin_pos_1_vs_2 = [2,8]
    begin_pos_2_vs_1 = [8,2]
    global big_job_in_porgress
    with big_job_in_porgress:
        big_job_waiter()
        for begins in begins_video:
            for x,y in zip(begin_pos_1_vs_2,begins):
                ffmpeg_VMAF_1_vs_2[x] = y
            job_1_vs_2 = ffmpeg_pool_big_job.apply_async(tools.launch_cmdExt, (ffmpeg_VMAF_1_vs_2,))
            
            for x,y in zip(begin_pos_2_vs_1,begins):
                ffmpeg_VMAF_2_vs_1[x] = y
            job_2_vs_1 = ffmpeg_pool_big_job.apply_async(tools.launch_cmdExt, (ffmpeg_VMAF_2_vs_1,))
            
            while len(out_1_vs_2) > 0:
                values_1_vs_2.append(float(re.search(r'.*\[Parsed_libvmaf.*\] VMAF score. (\d*.\d*).*',out_1_vs_2.pop()[1].decode("utf-8"), re.MULTILINE).group(1)))
                
            while len(out_2_vs_1) > 0:
                values_2_vs_1.append(float(re.search(r'.*\[Parsed_libvmaf.*\] VMAF score. (\d*.\d*).*',out_2_vs_1.pop()[1].decode("utf-8"), re.MULTILINE).group(1)))
    
            out_1_vs_2.append(job_1_vs_2.get())
            out_2_vs_1.append(job_2_vs_1.get())
    
    while len(out_1_vs_2) > 0:
        values_1_vs_2.append(float(re.search(r'.*\[Parsed_libvmaf.*\] VMAF score. (\d*.\d*).*',out_1_vs_2.pop()[1].decode("utf-8"), re.MULTILINE).group(1)))
            
    while len(out_2_vs_1) > 0:
        values_2_vs_1.append(float(re.search(r'.*\[Parsed_libvmaf.*\] VMAF score. (\d*.\d*).*',out_2_vs_1.pop()[1].decode("utf-8"), re.MULTILINE).group(1)))
    
    if mean(values_1_vs_2) >= mean(values_2_vs_1):
        return "1"
    else:
        return "2"
    
def get_good_frame(video_obj_1, video_obj_2, begin_in_sec, length_time, time_by_test, calculated_delay):
    import re
    from statistics import mean
    ffmpeg_PSNR = [tools.software["ffmpeg"], "-ss", "00:03:00", "-t", time_by_test, "-i", video_obj_1.filePath, 
       "-ss", "00:03:00", "-t", time_by_test, "-i", video_obj_2.filePath,
       "-lavfi", "[0:{}][1:{}]psnr".format(video_obj_1.video['StreamOrder'],video_obj_2.video['StreamOrder']),
       "-threads", str(tools.core_to_use), "-f", "null","-map", f"0:{video_obj_1.video['StreamOrder']}", "-map", f"1:{video_obj_2.video['StreamOrder']}", "-"]
    
    framerate_video_obj_1 = video_obj_1.get_fps()
    framerate_video_obj_2 = video_obj_2.get_fps()
    scale_video_obj_1 = video_obj_1.get_scale()
    scale_video_obj_2 = video_obj_2.get_scale()
    filter_modifications = []
    if framerate_video_obj_1 != None and framerate_video_obj_2 != None and framerate_video_obj_1 != framerate_video_obj_2:
        if framerate_video_obj_1 > framerate_video_obj_2:
            filter_modifications.append(f'fps=fps={framerate_video_obj_2}')
            frame_rate_use = framerate_video_obj_2
        else:
            filter_modifications.append(f'fps=fps={framerate_video_obj_1}')
            frame_rate_use = framerate_video_obj_1
    elif framerate_video_obj_1 != None:
        frame_rate_use = framerate_video_obj_1
    elif framerate_video_obj_2 != None:
        frame_rate_use = framerate_video_obj_2
    else:
        raise Exception(f"{video_obj_1.filePath} and {video_obj_2.filePath} have no framerate infos")
    if scale_video_obj_1 != None and scale_video_obj_2 != None and (scale_video_obj_1[0] != scale_video_obj_2[0] or scale_video_obj_1[1] != scale_video_obj_2[1]):
        if (scale_video_obj_1[0]*scale_video_obj_1[1]) > (scale_video_obj_2[0]*scale_video_obj_2[1]):
            filter_modifications.append(f'scale={scale_video_obj_1[0]}:{scale_video_obj_1[1]}')
        else:
            filter_modifications.append(f'scale={scale_video_obj_2[0]}:{scale_video_obj_2[1]}')
    if len(filter_modifications):
        ffmpeg_PSNR[13] = "-filter_complex"
        ffmpeg_PSNR[14] = "[0:{}]{}[0];[1:{}]{}[1]; [0][1]psnr".format(video_obj_1.video['StreamOrder'],", ".join(filter_modifications),video_obj_2.video['StreamOrder'],", ".join(filter_modifications))
    
    time_by_frame = 1.0/float(frame_rate_use)
    begin_in_sec_frame_adjusted = (float(int(begin_in_sec/time_by_frame))*time_by_frame)
    length_time_frame_adjusted = (float(int(length_time/time_by_frame))*time_by_frame)
    
    best_value_psnr = -1
    good_frame = -2
    global big_job_in_porgress
    with big_job_in_porgress:
        big_job_waiter()
        for i in range(-2,3):
            jobs_psnr = []
            for begins in generate_cut_to_compare_video_quality(begin_in_sec_frame_adjusted,(float(int((begin_in_sec_frame_adjusted + calculated_delay)/time_by_frame)+i)*time_by_frame),length_time_frame_adjusted):
                ffmpeg_PSNR[2] = begins[0]
                ffmpeg_PSNR[8] = begins[1]
                jobs_psnr.append(ffmpeg_pool_big_job.apply_async(tools.launch_cmdExt, (ffmpeg_PSNR,)))
            
            '''
                TODO:
                    Create thread for process all time.
            '''
            list_result_average_psnr = []
            for job_psnr in jobs_psnr:
                result = job_psnr.get()
                list_result_average_psnr.append(float(re.search(r'.*\[Parsed_psnr.*\].+average:(\d+.\d+).*',result[1].decode("utf-8"), re.MULTILINE).group(1)))
            
            if mean(list_result_average_psnr) >= best_value_psnr:
                good_frame = i
                best_value_psnr = mean(list_result_average_psnr)
    
    calculated_delay = (float(int((begin_in_sec_frame_adjusted + calculated_delay)/time_by_frame)+good_frame)*time_by_frame) - begin_in_sec_frame_adjusted
    return calculated_delay*1000,generate_cut_to_compare_video_quality(begin_in_sec_frame_adjusted,(float(int((begin_in_sec_frame_adjusted + calculated_delay)/time_by_frame)+good_frame)*time_by_frame),length_time_frame_adjusted)

def get_common_audios_language(videosObj):
    commonLanguages = set(videosObj[0].audios.keys())
    for videoObj in videosObj:
        commonLanguages = commonLanguages.intersection(videoObj.audios.keys())
    return commonLanguages

def get_worse_quality_audio_param(videosObj,language,rules):
    try:
        worseAudio = [0,0]
        while language not in videosObj[worseAudio[0]].audios and len(videosObj) > worseAudio[0]:
            worseAudio[0]+=1
        if len(videosObj[worseAudio[0]].audios[language]) > 1:
            for j in range(1,len(videosObj[worseAudio[0]].audios[language])):
                if (not test_if_the_best_by_rules_audio_entry(videosObj[worseAudio[0]].audios[language][worseAudio[1]],videosObj[worseAudio[0]].audios[language][j],rules)):
                    worseAudio[1] = j
        if len(videosObj) > worseAudio[0]+1:
            for i in range(worseAudio[0]+1,len(videosObj)):
                for j in range(0,len(videosObj[i].audios[language])):
                    if (not test_if_the_best_by_rules_audio_entry(videosObj[worseAudio[0]].audios[language][worseAudio[1]],videosObj[i].audios[language][j],rules)):
                        worseAudio = [i,j]
        
        if 'BitRate' not in videosObj[worseAudio[0]].audios[language][worseAudio[1]]:
            videosObj[worseAudio[0]].audios[language][worseAudio[1]]['BitRate'] = videosObj[worseAudio[0]].audios[language][worseAudio[1]]['BitRate_Nominal']
        return videosObj[worseAudio[0]].audios[language][worseAudio[1]].copy()
    except:
        return {'Format':"MP3",
                'Channels':"2",
                'BitRate':"128000",
                'SamplingRate':"44100"}

def get_less_channel_number(videos_obj,language):
    try:
        less_channel_number = [0,0]
        while language not in videos_obj[less_channel_number[0]].audios and len(videos_obj) > less_channel_number[0]:
            less_channel_number[0]+=1
        
        if len(videos_obj[less_channel_number[0]].audios[language]) > 1:
            for j in range(1,len(videos_obj[less_channel_number[0]].audios[language])):
                if int(videos_obj[less_channel_number[0]].audios[language][less_channel_number[1]]['Channels']) > int(videos_obj[less_channel_number[0]].audios[language][j]['Channels']):
                    less_channel_number[1] = j
        if len(videos_obj) > less_channel_number[0]+1:
            for i in range(less_channel_number[0]+1,len(videos_obj)):
                for j in range(0,len(videos_obj[i].audios[language])):
                    if int(videos_obj[less_channel_number[0]].audios[language][less_channel_number[1]]['Channels']) > int(videos_obj[i].audios[language][j]['Channels']):
                        less_channel_number = [i,j]

        return videos_obj[less_channel_number[0]].audios[language][less_channel_number[1]]['Channels']
    except:
        return "2"

def get_shortest_audio_durations(videosObj,language):
    shorter = 1000000000000000000000000000000000
    for videoObj in videosObj:
        for audio in videoObj.audios[language]:
            if float(audio['Duration']) < shorter:
                shorter = float(audio['Duration'])
    return shorter

def get_shortest_video_durations(videosObj):
    shorter = 1000000000000000000000000000000000
    for videoObj in videosObj:
        video_duration = videoObj.get_video_duration()
        if video_duration < shorter:
            shorter = video_duration
    return shorter

def get_birate_key(data):
    if 'BitRate' in data:
        return 'BitRate'
    elif 'BitRate_Nominal' in data:
        return 'BitRate_Nominal'
    else:
        raise Exception(f"No video bitrate {data}")
    
def get_bitrate(data):
    return data[get_birate_key(data)]

def test_if_the_best_by_rules_video_entry(base,challenger,rules):
    if base['Encoded_Library_Name'] == challenger['Encoded_Library_Name']:
        return float(base[get_birate_key(base)]) < float(challenger[get_birate_key(challenger)])*(1+(0.05*(float(challenger['Format_Level'])-float(base['Format_Level']))))
    else:
        return test_if_the_best_by_rules(base['Encoded_Library_Name'],base[get_birate_key(base)],challenger['Encoded_Library_Name'],challenger[get_birate_key(challenger)],rules)

def test_if_the_best_by_rules_audio_entry(base,challenger,rules):
    if base['Format'] == challenger['Format']:
        return base['BitRate'] < challenger['BitRate']
    else:
        return test_if_the_best_by_rules(base['Format'],get_birate_key(base),challenger['Format'],get_birate_key(base),rules)
    
def test_if_the_best_by_rules(formatFileBase,bitrateFileBase,formatFileChallenger,bitrateFileChallenger,rules,inEgualityKeepChallenger=False):
    testResul = test_if_it_better_by_rules(formatFileBase.lower(),bitrateFileBase,formatFileChallenger.lower(),bitrateFileChallenger,rules)
    if testResul == 2:
        return inEgualityKeepChallenger
    else:
        return testResul

'''
Test if we have a better format
Return:
    0/False : The base is the best
    1/True : The challenger is the best
    2 : The two are good
    '''
def test_if_it_better_by_rules(formatFileBase,bitrateFileBase,formatFileChallenger,bitrateFileChallenger,rules):
    if formatFileBase in rules and formatFileChallenger in rules:
        if formatFileBase in rules[formatFileChallenger]:
            if isinstance(rules[formatFileChallenger][formatFileBase], float):
                ponderateBitrateChallenger = float(bitrateFileChallenger)*rules[formatFileChallenger][formatFileBase]
                if ponderateBitrateChallenger > float(bitrateFileBase):
                    return True
                elif ponderateBitrateChallenger < float(bitrateFileBase):
                    return False
                elif formatFileBase == formatFileChallenger and bitrateFileBase > bitrateFileChallenger:
                    return False
                elif formatFileBase == formatFileChallenger and bitrateFileBase < bitrateFileChallenger:
                    return True
                else:
                    return 2
            else:
                return rules[formatFileChallenger][formatFileBase]
        else:
            return 2
    else:
        return 2

def md5_calculator(filePath,streamID,start_time=0,end_time=None,duration_stream=None):
    cmd = [
    tools.software["ffmpeg"], "-v", "error", "-i", filePath,
    "-ss", str(start_time)]

    if end_time != None:
        if duration_stream != None:
            if duration_stream > end_time:
                cmd.extend(["-t", str(end_time-start_time)])
            else:
                cmd.extend(["-t", str(duration_stream-start_time)])
        else:
            cmd.extend(["-t", str(end_time-start_time)])

    cmd.extend(["-map", f"0:{streamID}", "-c", "copy", "-f", "md5", "-"
    ])
    stdout, stderror, exitCode = tools.launch_cmdExt(cmd)
    if exitCode == 0:
        md5 = stdout.decode("utf-8").strip().split("=")[-1]
        return (streamID, md5)
    return (streamID, None)

def subtitle_text_md5(filePath,streamID):
    number_of_style = count_font_lines_in_ass(filePath, streamID)
    if number_of_style == None or number_of_style > 1:
        return subtitle_text_ass_md5(filePath,streamID)
    else:
        return subtitle_text_srt_md5(filePath,streamID)

def subtitle_text_srt_md5(filePath,streamID):
    import hashlib
    import re
    cmd = [
        tools.software["ffmpeg"], "-v", "error", "-threads", str(tools.core_to_use), "-i", filePath,
        "-map", f"0:{streamID}",
         "-c:s", "srt",
        "-f", "srt", "pipe:1"
    ]
    stdout, stderror, exitCode = tools.launch_cmdExt(cmd)
    if exitCode == 0:
        lines = stdout.decode('utf-8', errors='ignore').splitlines()
        text_lines = [re.sub(r'<[^<]+>', '', line) for line in lines if line.strip() and (not line.strip().isdigit()) and ("-->" not in line)]
        filtered_text = "\n".join(text_lines).encode('utf-8')
        md5 = hashlib.md5(filtered_text).hexdigest()
        if (not text_lines):
            stderr.write(f"No subtitle text found in {filePath}, stream {streamID}\n")
            return (streamID, None)
        else:
            return (streamID, md5)
    else:
        return (streamID, None)

def count_font_lines_in_ass(filePath, streamID):
    import re
    cmd = [
        "ffmpeg",
        "-v", "error", "-threads", str(tools.core_to_use),
        "-i", filePath,
        "-map", f"0:{streamID}",
        "-c:s", "ass",
        "-f", "ass",
        "pipe:1"
    ]
    
    stdout, stderror, exitCode = tools.launch_cmdExt(cmd)
    if exitCode == 0:
        lines = stdout.decode('utf-8', errors='ignore').splitlines()

        style_pattern = re.compile(r'^Style:.+', re.IGNORECASE)
        count = sum(1 for line in lines if style_pattern.match(line))

        return count
    else:
        stderr.write(f"Error extracting ASS from {filePath}, stream {streamID}, with error {stderror.decode('utf-8', errors='ignore')} \n")
        return None

def subtitle_text_ass_md5(filePath,streamID):
    import hashlib
    import re
    cmd = [
        tools.software["ffmpeg"], "-v", "error", "-threads", str(tools.core_to_use), "-i", filePath,
        "-map", f"0:{streamID}",
         "-c:s", "ass",
        "-f", "ass", "pipe:1"
    ]
    stdout, stderror, exitCode = tools.launch_cmdExt(cmd)
    if exitCode == 0:
        lines = stdout.decode('utf-8', errors='ignore').splitlines()
        text_lines = [re.sub(r'^[^,\n]+,\d[^,\n]+,[^,\n]+,', '', line) for line in lines if line.strip()]
        filtered_text = "\n".join(text_lines).encode('utf-8')
        md5 = hashlib.md5(filtered_text).hexdigest()
        if (not text_lines):
            stderr.write(f"No subtitle text found in {filePath}, stream {streamID}\n")
            return (streamID, None)
        else:
            return (streamID, md5)
    else:
        return (streamID, None)