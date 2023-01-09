'''
Created on 23 Apr 2022

@author: francois
'''

from os import path,remove
import tools
import json

ffmpeg_pool = None

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
        for data in self.mediadata['media']['track']:
            if data['@type'] == 'Video':
                if self.video != None:
                    raise Exception(self.filePath+" multiple video in same file")
                self.video = data
            elif data['@type'] == 'Audio':
                if 'Language' in data:
                    language = data['Language']
                else:
                    language = "und"
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
            self.audios[tools.default_language_for_undetermine] = self.audios["und"]
            
    def get_fps(self):
        if 'FrameRate' in self.video:
            return float(self.video['FrameRate'])
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

def get_best_quality_video(video_obj_1, video_obj_2, begins_video, time_by_test):
    import re
    from statistics import mean
    ffmpeg_VMAF_1_vs_2 = [tools.software["ffmpeg"], "-ss", "00:03:00", "-t", time_by_test, "-i", video_obj_1.filePath, 
           "-ss", "00:03:00", "-t", time_by_test, "-i", video_obj_2.filePath,
           "-lavfi", "libvmaf=n_threads={}:log_fmt=json".format(tools.core_to_use),
           "-threads", str(tools.core_to_use), "-f", "null", "-"]
    
    framerate_video_obj_1 = video_obj_1.get_fps()
    framerate_video_obj_2 = video_obj_2.get_fps()
    if framerate_video_obj_1 != None and framerate_video_obj_2 != None and framerate_video_obj_1 != framerate_video_obj_2:
        ffmpeg_VMAF_1_vs_2[13] = "-filter_complex"
        if framerate_video_obj_1 > framerate_video_obj_2:
            ffmpeg_VMAF_1_vs_2[14] = "[0:v]fps=fps={}[0];[1:v]fps=fps={}[1]; [0][1]libvmaf=n_threads={}:log_fmt=json".format(framerate_video_obj_2,framerate_video_obj_2,tools.core_to_use)
        else:
            ffmpeg_VMAF_1_vs_2[14] = "[0:v]fps=fps={}[0];[1:v]fps=fps={}[1]; [0][1]libvmaf=n_threads={}:log_fmt=json".format(framerate_video_obj_1,framerate_video_obj_1,tools.core_to_use)

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
            for j in range(0,len(videosObj[worseAudio[0]].audios[language])):
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