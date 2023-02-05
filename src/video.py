'''
Created on 23 Apr 2022

@author: francois
'''

from os import path,remove
from sys import stderr
from time import strftime,gmtime
import tools
import json

ffmpeg_pool = None
path_to_livmaf_model = "" #Nothing if it use the default
number_cut = 5
percent_time_by_test_video_quality_from_cut = 25

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
        self.audios = None
        self.commentary = None
        self.video = None
        self.subtitles = None
        self.video_quality = None
        self.tmpFiles = {}
        self.ffmpeg_progress = []
        self.delays = {}
    
    def get_mediadata(self):
        stdout, stderror, exitCode = tools.launch_cmdExt([tools.software["mediainfo"], "--Output=JSON", self.filePath])
        if exitCode != 0:
            raise Exception("Error with {} during the mediadata: {}".format(self.filePath,stderror.decode("UTF-8")))
        self.mediadata = json.loads(stdout.decode("UTF-8"))
        self.audios = {}
        self.subtitles = {}
        self.commentary = {}
        for data in self.mediadata['media']['track']:
            if data['@type'] == 'Video':
                if self.video != None:
                    raise Exception(f"Multiple video in the same file {self.filePath}, I can't compare and merge they")
                else:
                    self.video = data
            elif data['@type'] == 'Audio':
                if 'Language' in data:
                    language = data['Language']
                else:
                    language = "und"
                if ('Title' in data and 'Commentary' == data['Title']):
                    if language in self.commentary:
                        self.commentary[language].append(data)
                    else:
                        self.commentary[language] = [data]
                else:
                    if language in self.audios:
                        self.audios[language].append(data)
                    else:
                        self.audios[language] = [data]
            elif data['@type'] == 'Text':
                if 'Language' in data:
                    if data['Language'] in self.subtitles:
                        self.subtitles[data['Language']].append(data)
                    else:
                        self.subtitles[data['Language']] = [data]
        if "und" in self.audios and len(self.audios) == 1:
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
    
    def extract_audio_in_part(self,language,exportParam,cutTime=None):
        global ffmpeg_pool
        nameFilesExtract = []
        if 'audio' in self.tmpFiles:
            self.remove_tmp_files(type_file="audio")
        self.tmpFiles['audio'] = nameFilesExtract

        baseCommand = [tools.software["ffmpeg"], "-y", "-threads", str(tools.core_to_use), "-nostdin", "-i", self.filePath, "-vn", "-acodec", exportParam["Format"].lower().replace('-',''), "-ab", exportParam["BitRate"], "-ar", exportParam['SamplingRate']]
        if exportParam['Channels'] == "2":
            baseCommand.extend(["-ac", exportParam['Channels']])
        if cutTime == None:
            nameFilesExtractCut = []
            nameFilesExtract.append(nameFilesExtractCut)
            for audio in self.audios[language]:
                nameOutFile = path.join(tools.tmpFolder,self.fileBaseName+"."+str(audio["ID"])+".1"+"."+exportParam['Format'].lower().replace('-',''))
                nameFilesExtractCut.append(nameOutFile)
                cmd = baseCommand.copy()
                cmd.extend(["-map", "0:"+str(int(audio["ID"])-1), nameOutFile])
                self.ffmpeg_progress.append(ffmpeg_pool.apply_async(tools.launch_cmdExt, (cmd,)))
        else:
            for audio in self.audios[language]:
                nameFilesExtractCut = []
                nameFilesExtract.append(nameFilesExtractCut)
                cutNumber = 0
                for cut in cutTime:
                    nameOutFile = path.join(tools.tmpFolder,self.fileBaseName+"."+str(audio["ID"])+"."+str(cutNumber)+"."+exportParam['Format'].lower().replace('-',''))
                    nameFilesExtractCut.append(nameOutFile)
                    cmd = baseCommand.copy()
                    cmd.extend(["-map", "0:"+str(int(audio["ID"])-1), "-ss", cut[0], "-t", cut[1] , nameOutFile])
                    self.ffmpeg_progress.append(ffmpeg_pool.apply_async(tools.launch_cmdExt, (cmd,)))
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
                
    def wait_end_ffmpeg_progress(self):
        while len(self.ffmpeg_progress) > 0:
            ffmpeg_job = self.ffmpeg_progress.pop(0)
            ffmpeg_job.get()

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

def get_best_quality_video(video_obj_1, video_obj_2, begins_video, time_by_test):
    import re
    from statistics import mean
    ffmpeg_VMAF_1_vs_2 = [tools.software["ffmpeg"], "-ss", "00:03:00", "-t", time_by_test, "-i", video_obj_1.filePath, 
           "-ss", "00:03:00", "-t", time_by_test, "-i", video_obj_2.filePath,
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
    for begins in begins_video:
        for x,y in zip(begin_pos_1_vs_2,begins):
            ffmpeg_VMAF_1_vs_2[x] = y
        job_1_vs_2 = ffmpeg_pool.apply_async(tools.launch_cmdExt, (ffmpeg_VMAF_1_vs_2,))
        
        for x,y in zip(begin_pos_2_vs_1,begins):
            ffmpeg_VMAF_2_vs_1[x] = y
        job_2_vs_1 = ffmpeg_pool.apply_async(tools.launch_cmdExt, (ffmpeg_VMAF_2_vs_1,))
        
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
    for i in range(-2,3):
        jobs_psnr = []
        for begins in generate_cut_to_compare_video_quality(begin_in_sec_frame_adjusted,(float(int((begin_in_sec_frame_adjusted + calculated_delay)/time_by_frame)+i)*time_by_frame),length_time_frame_adjusted):
            ffmpeg_PSNR[2] = begins[0]
            ffmpeg_PSNR[8] = begins[1]
            jobs_psnr.append(ffmpeg_pool.apply_async(tools.launch_cmdExt, (ffmpeg_PSNR,)))
        
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
    return videosObj[worseAudio[0]].audios[language][worseAudio[1]].copy()

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

def test_if_the_best_by_rules_video_entry(base,challenger,rules):
    if base['Encoded_Library_Name'] == challenger['Encoded_Library_Name']:
        return float(base[get_birate_key(base)]) < float(challenger[get_birate_key(challenger)])*(1+(0.05*(float(challenger['Format_Level'])-float(base['Format_Level']))))
    else:
        return test_if_the_best_by_rules(base['Encoded_Library_Name'],base[get_birate_key(base)],challenger['Encoded_Library_Name'],challenger[get_birate_key(challenger)],rules)

def test_if_the_best_by_rules_audio_entry(base,challenger,rules):
    if base['Format'] == challenger['Format']:
        return base['BitRate'] < challenger['BitRate']
    else:
        return test_if_the_best_by_rules(base['Format'],base['BitRate'],challenger['Format'],challenger['BitRate'],rules)
    
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