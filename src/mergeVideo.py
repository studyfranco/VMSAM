'''
Created on 24 Apr 2022

@author: studyfranco

This software need libchromaprint-tools,ffmpeg,mediainfo
'''

import argparse
import sys
from datetime import datetime
from multiprocessing import Pool
from os import path,remove,chdir
from random import shuffle
from statistics import variance,mean
from threading import Thread
from time import strftime,gmtime

numberCut = 5
max_delay_variance_second_method = 0.005
cut_file_to_get_delay_second_method = 2.5 # With the second method we need a better result. After we check the two file is compatible, we need a serious right result adjustment
percent_time_by_test_video_quality_from_cut = 25

def decript_merge_rules(stringRules):
    rules = {}
    egualRules = set()
    besterBy = []
    for subRules in stringRules.split(","):
        bester = None
        precedentSuperior = []
        for subSubRules in subRules.split(">"):
            if '*' in subSubRules:
                value,multValue = subSubRules.lower().split("*")
                multValue = float(multValue)
            else:
                value = subSubRules.lower()
                multValue = True
            value = value.split("=")
            for subValue in value:
                if subValue not in rules:
                    rules[subValue] = {}
            for sup in precedentSuperior:
                if sup[0] == subValue:
                    pass
                elif isinstance(sup[1], float):
                    for subValue in value:
                        if subValue not in rules[sup[0]] or (isinstance(rules[sup[0]][subValue], float) and rules[sup[0]][subValue] > 1 and rules[sup[0]][subValue] < sup[1]):
                            rules[sup[0]][subValue] = sup[1]
                            rules[subValue][sup[0]] = 1.0/sup[1]
                elif isinstance(sup[1], bool):
                    for subValue in value:
                        rules[sup[0]][subValue] = sup[1]
                        rules[subValue][sup[0]] = (not sup[1])
                    
                if isinstance(multValue, bool):
                    sup[1] = multValue
                elif isinstance(sup[1], float):
                    sup[1] = sup[1]*multValue
                    
            for subValue in value:
                precedentSuperior.append([subValue,multValue])
                for subValue2 in value:
                    if subValue2 != subValue:
                        egualRules.add((subValue,subValue2))
                        egualRules.add((subValue2,subValue))
                        
            if bester != None:
                for best in bester:
                    for subValue in value:
                        besterBy.append([best,subValue])
            
            if isinstance(multValue, bool) and multValue:
                bester = value
            else:
                bester = None
    
    for besterRules in besterBy:
        decript_merge_rules_bester(rules,besterRules[0],besterRules[1])
    
    for egualRule in egualRules:
        if egualRule[1] in rules[egualRule[0]]:
            del rules[egualRule[0]][egualRule[1]]
    
    return rules

def decript_merge_rules_bester(rules,best,weak):
    for rulesWeak in rules[weak].items():
        if (isinstance(rulesWeak[1], bool) and rulesWeak[1]) or (isinstance(rulesWeak[1], float) and rulesWeak[1] > 5):
            decript_merge_rules_bester(rules,best,rulesWeak[0])
    rules[weak][best] = False
    rules[best][weak] = True
    
def get_good_parameters_to_get_fidelity(videosObj,language,audioParam,maxTime):
    if maxTime < 10:
        timeTake = strftime('%H:%M:%S',gmtime(maxTime))
    else:
        timeTake = "00:00:10"
    for videoObj in videosObj:
        videoObj.extract_audio_in_part(language,audioParam,cutTime=[["00:00:00",timeTake]])
        videoObj.wait_end_ffmpeg_progress()
        if (not test_calcul_can_be(videoObj.tmpFiles['audio'][0][0])):
            audioParam["Format"] = "AAC"
            audioParam['Channels'] = "2"
            if int(audioParam["BitRate"]) > 256000:
                audioParam["BitRate"] = "256000"
            videoObj.extract_audio_in_part(language,audioParam,cutTime=[["00:00:00",timeTake]])
            videoObj.wait_end_ffmpeg_progress()
            if (not test_calcul_can_be(videoObj.tmpFiles['audio'][0][0])):
                audioParam["Format"] = "MP3"
            else:
                raise Exception(f"Audio parameters to get the fidelity not working with {videoObj.filePath}")

def get_delay_fidelity(video_obj_1,video_obj_2,lenghtTime,ignore_audio_couple=set()):
    global numberCut
    delay_Fidelity_Values = {}
    video_obj_1.wait_end_ffmpeg_progress()
    video_obj_2.wait_end_ffmpeg_progress()
    for i in range(0,len(video_obj_1.tmpFiles['audio'])):
        for j in range(0,len(video_obj_2.tmpFiles['audio'])):
            if f"{i}-{j}" not in ignore_audio_couple:
                delay_between_two_audio = []
                delay_Fidelity_Values[f"{i}-{j}"] = delay_between_two_audio
                for h in range(0,numberCut):
                    delay_between_two_audio.append(correlate(video_obj_1.tmpFiles['audio'][i][h],video_obj_2.tmpFiles['audio'][j][h],lenghtTime))
    return delay_Fidelity_Values

def get_delay_by_second_method(video_obj_1,video_obj_2,ignore_audio_couple=set()):
    global numberCut
    delay_Values = {}
    video_obj_1.wait_end_ffmpeg_progress()
    video_obj_2.wait_end_ffmpeg_progress()
    for i in range(0,len(video_obj_1.tmpFiles['audio'])):
        for j in range(0,len(video_obj_2.tmpFiles['audio'])):
            if f"{i}-{j}" not in ignore_audio_couple:
                delay_between_two_audio = []
                delay_Values[f"{i}-{j}"] = delay_between_two_audio
                for h in range(0,numberCut):
                    delay_between_two_audio.append(second_correlation(video_obj_1.tmpFiles['audio'][i][h],video_obj_2.tmpFiles['audio'][j][h]))
    return delay_Values

def get_begin_time_with_millisecond(delay,beginInSecBeforeDelay):
    begining_in_second = beginInSecBeforeDelay
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

class compare_video(Thread):
    '''
    classdocs
    '''


    def __init__(self, video_obj_1,video_obj_2,beginInSec,audioParam,language,lenghtTime,lenghtTimePrepare):
        '''
        Constructor
        '''
        Thread.__init__(self)
        self.video_obj_1 = video_obj_1
        self.video_obj_2 = video_obj_2
        self.beginInSec = beginInSec
        self.audioParam = audioParam
        self.language = language
        self.lenghtTime = lenghtTime
        self.lenghtTimePrepare = lenghtTimePrepare
        self.video_obj_with_best_quality = None

    def run(self):
        try:
            delay = self.test_if_constant_good_delay()
            self.get_best_video(delay)
        except Exception as e:
            sys.stderr.write(str(e)+"\n")
        
    def test_if_constant_good_delay(self):
        delay_first_method,ignore_audio_couple = self.first_delay_test()
        self.recreate_files_for_delay_adjuster(delay_first_method-500)
        delay_second_method = self.second_delay_test(delay_first_method-500,ignore_audio_couple)
        
        self.recreate_cut_files_audio_first_file()
        calculated_delay = delay_first_method+round(delay_second_method*1000)-500
        if abs(calculated_delay-delay_first_method) < 500:
            self.recreate_files_for_delay_adjuster(calculated_delay)
            return calculated_delay
        else:
            self.recreate_files_for_delay_adjuster(0)
            raise Exception(f"Delay found between {self.video_obj_1.filePath} and {self.video_obj_2.filePath} is unexpected between the two methods")
        
    def first_delay_test(self):
        delay_Fidelity_Values = get_delay_fidelity(self.video_obj_1,self.video_obj_2,self.lenghtTime)
        ignore_audio_couple = set()
        delay_detected = set()
        for key_audio, delay_fidelity_list in delay_Fidelity_Values.items():
            set_delay = set()
            for delay_fidelity in delay_fidelity_list:
                set_delay.add(delay_fidelity[2])
            if len(set_delay) == 1:
                delay_detected.update(set_delay)
            else:
                # Work in progress
                # We need to ask to the user to pass them if they want.
                ignore_audio_couple.add(key_audio)
        
        if len(delay_detected) != 1:
            raise Exception(f"Multiple delay found with {self.video_obj_1.filePath} and {self.video_obj_2.filePath}")
        else:
            delayUse = list(delay_detected)[0]
        
        self.recreate_files_for_delay_adjuster(delayUse)
        
        delay_Fidelity_Values = get_delay_fidelity(self.video_obj_1,self.video_obj_2,self.lenghtTime,ignore_audio_couple=ignore_audio_couple)
        delay_detected = set()
        for key_audio, delay_fidelity_list in delay_Fidelity_Values.items():
            set_delay = set()
            for delay_fidelity in delay_fidelity_list:
                set_delay.add(delay_fidelity[2])
            if len(set_delay) == 1:
                delay_detected.update(set_delay)
            else:
                raise Exception(f"Multiple delay found with {self.video_obj_1.filePath} and {self.video_obj_2.filePath}")
                    
        if len(delay_detected) == 1 and 0 in delay_detected:
            return delayUse,ignore_audio_couple
        else:
            delayUse += list(delay_detected)[0]
            self.recreate_files_for_delay_adjuster(delayUse)
            
            delay_Fidelity_Values = get_delay_fidelity(self.video_obj_1,self.video_obj_2,self.lenghtTime,ignore_audio_couple=ignore_audio_couple)
            delay_detected = set()
            for key_audio, delay_fidelity_list in delay_Fidelity_Values.items():
                set_delay = set()
                for delay_fidelity in delay_fidelity_list:
                    set_delay.add(delay_fidelity[2])
                if len(set_delay) == 1:
                    delay_detected.update(set_delay)
                else:
                    raise Exception(f"Multiple delay found with {self.video_obj_1.filePath} and {self.video_obj_2.filePath}")
                        
            if len(delay_detected) == 1 and 0 in delay_detected:
                return delayUse,ignore_audio_couple
            else:
                raise Exception(f"Not able to find delay for {self.video_obj_1} and {self.video_obj_2}")
        
    def recreate_files_for_delay_adjuster(self,delayUse):
        cutTime = []
        begining_in_second, begining_in_millisecond = get_begin_time_with_millisecond(delayUse,self.beginInSec)
        for i in range(0,numberCut):
            cutTime.append([strftime('%H:%M:%S',gmtime(begining_in_second))+begining_in_millisecond,self.lenghtTimePrepare])
            begining_in_second += self.lenghtTime
        
        self.video_obj_2.extract_audio_in_part(self.language,self.audioParam,cutTime=cutTime)
        
    def second_delay_test(self,delayUse,ignore_audio_couple):
        global max_delay_variance_second_method
        global cut_file_to_get_delay_second_method
        global numberCut
        delay_Values = get_delay_by_second_method(self.video_obj_1,self.video_obj_2,ignore_audio_couple=ignore_audio_couple)
        delay_detected = set()
        for key_audio, delay_list in delay_Values.items():
            set_delay = set()
            for delay in delay_list:
                set_delay.add(delay[1])
            if variance(set_delay) < max_delay_variance_second_method:
                delay_detected.update(set_delay)
            else:
                raise Exception(f"Variance delay in the second test is to big with {self.video_obj_1.filePath} and {self.video_obj_2.filePath}")
        
        if variance(delay_detected) > max_delay_variance_second_method:
            raise Exception(f"Multiple delay found with {self.video_obj_1.filePath} and {self.video_obj_2.filePath}")
        else:
            self.video_obj_1.extract_audio_in_part(self.language,self.audioParam,cutTime=[[strftime('%H:%M:%S',gmtime(self.beginInSec)),strftime('%H:%M:%S',gmtime(int(self.lenghtTime*numberCut/cut_file_to_get_delay_second_method)))]])
            begining_in_second, begining_in_millisecond = get_begin_time_with_millisecond(delayUse,self.beginInSec)
            self.video_obj_2.extract_audio_in_part(self.language,self.audioParam,cutTime=[[strftime('%H:%M:%S',gmtime(begining_in_second))+begining_in_millisecond,strftime('%H:%M:%S',gmtime(int(self.lenghtTime*numberCut/cut_file_to_get_delay_second_method)))]])
            self.video_obj_1.wait_end_ffmpeg_progress()
            self.video_obj_2.wait_end_ffmpeg_progress()
            for i in range(0,len(self.video_obj_1.tmpFiles['audio'])):
                for j in range(0,len(self.video_obj_2.tmpFiles['audio'])):
                    if f"{i}-{j}" not in ignore_audio_couple:
                        delay_between_two_audio = []
                        delay_Values[f"{i}-{j}"] = delay_between_two_audio
                        delay_between_two_audio.append(second_correlation(self.video_obj_1.tmpFiles['audio'][i][0],self.video_obj_2.tmpFiles['audio'][j][0]))
            
            import gc
            gc.collect()
            delay_detected = set()
            for key_audio, delay_list in delay_Values.items():
                set_delay = set()
                for delay in delay_list:
                    set_delay.add(delay[1])
                delay_detected.update(set_delay)
            return mean(delay_detected)
    
    def recreate_cut_files_audio_first_file(self):
        cutTime = []
        lenghtTimePrepare = strftime('%H:%M:%S',gmtime(self.lenghtTime))
        beginInSecToModify = self.beginInSec
        for i in range(0,numberCut):
            cutTime.append([strftime('%H:%M:%S',gmtime(self.beginInSec)),lenghtTimePrepare])
            beginInSecToModify += self.lenghtTime
            
    def get_best_video(self,delay):
        if percent_time_by_test_video_quality_from_cut >= 100:
            time_by_test = self.lenghtTime
        else:
            time_by_test = int(self.lenghtTime*percent_time_by_test_video_quality_from_cut/100)+1
        
        begins_video = []
        begin_in_second_video_1 = self.beginInSec
        begin_in_second_video_2, begin_in_millisecond_video_2 = get_begin_time_with_millisecond(delay,self.beginInSec)
        for i in range(0,numberCut):
            begins_video.append([strftime('%H:%M:%S',gmtime(begin_in_second_video_1)),strftime('%H:%M:%S',gmtime(begin_in_second_video_2))+begin_in_millisecond_video_2])
            begin_in_second_video_1 += self.lenghtTime
            begin_in_second_video_2 += self.lenghtTime
        
        if video.get_best_quality_video(self.video_obj_1, self.video_obj_2, begins_video, strftime('%H:%M:%S',gmtime(time_by_test))) == 1:
            self.video_obj_with_best_quality = self.video_obj_1
        else:
            self.video_obj_with_best_quality = self.video_obj_2
        
def get_delay_and_best_video(videosObj,language,audioRules):
    worseAudioQualityWillUse = video.get_worse_quality_audio_param(videosObj,language,audioRules)
    minAudioTimeInSec = video.get_shortest_audio_durations(videosObj,language)
    get_good_parameters_to_get_fidelity(videosObj,language,worseAudioQualityWillUse,minAudioTimeInSec)
    if minAudioTimeInSec > 540:
        beginInSec = 120
        lenghtTime = int((minAudioTimeInSec-240)/numberCut)
    elif minAudioTimeInSec > 60:
        beginInSec = 30
        lenghtTime = int((minAudioTimeInSec-45)/numberCut)
    elif minAudioTimeInSec > 5:
        beginInSec = 0
        lenghtTime = int(minAudioTimeInSec-2/numberCut)
    cutTime = []
    lenghtTimePrepare = strftime('%H:%M:%S',gmtime(lenghtTime))
    beginInSecOriginal = beginInSec
    for i in range(0,numberCut):
        cutTime.append([strftime('%H:%M:%S',gmtime(beginInSec)),lenghtTimePrepare])
        beginInSec += lenghtTime
    for videoObj in videosObj:
        videoObj.extract_audio_in_part(language,worseAudioQualityWillUse,cutTime=cutTime)
        
    dict_file_path_obj = {}
    for videoObj in videosObj:
        dict_file_path_obj[videoObj.filePath] = videoObj

    compareObjs = videosObj.copy()
    already_compared = {}
    already_compared_with_result = set()
    already_compared_without_result = set()
    list_not_compatible_video = []
    while len(compareObjs) > 1:
        if len(compareObjs)%2 != 0:
            new_compare_objs = [compareObjs.pop()]
        else:
            new_compare_objs = []
        list_in_compare_video = []
        for i in range(0,len(compareObjs),2):
            nameInList = [compareObjs[i].filePath,compareObjs[i+1].filePath]
            sorted(nameInList)
            if nameInList[0] not in already_compared or (nameInList[0] in already_compared and nameInList[1] not in already_compared[nameInList[0]]):
                list_in_compare_video.append(compare_video(compareObjs[i],compareObjs[i+1],beginInSecOriginal,worseAudioQualityWillUse,language,lenghtTime,lenghtTimePrepare))
                list_in_compare_video[-1].start()
            else:
                new_compare_objs.append(compareObjs[i])
                new_compare_objs.append(compareObjs[i+1])
        
        compareObjs = new_compare_objs
        for compare_video_obj in list_in_compare_video:
            nameInList = [compare_video_obj.video_obj_1.filePath,compare_video_obj.video_obj_2.filePath]
            sorted(nameInList)
            compare_video_obj.join()
            if compare_video_obj.video_obj_with_best_quality != None:
                valueObtain = compare_video_obj.video_obj_with_best_quality.filePath==nameInList[0]
                compareObjs.append(compare_video_obj.video_obj_with_best_quality)
                already_compared_with_result.add(compare_video_obj.video_obj_1.filePath)
                already_compared_with_result.add(compare_video_obj.video_obj_2.filePath)
                tools.remove_element_without_bug(already_compared_without_result,compare_video_obj.video_obj_1.filePath)
                tools.remove_element_without_bug(already_compared_without_result,compare_video_obj.video_obj_2.filePath)
            else:
                valueObtain = None
                compareObjs.append(compare_video_obj.video_obj_1)
                compareObjs.append(compare_video_obj.video_obj_2)
                already_compared_without_result.add(compare_video_obj.video_obj_1.filePath)
                already_compared_without_result.add(compare_video_obj.video_obj_2.filePath)
            if nameInList[0] in already_compared:
                already_compared[nameInList[0]][nameInList[1]] = valueObtain
            else:
                already_compared[nameInList[0]] = {nameInList[1]: valueObtain}
            
        shuffle(compareObjs)
        
    
    
def mergeVideo(files,outFolder,inFolder=None):
    videosObj = []
    name_file = {}
    if inFolder == None:
        for file in files:
            videosObj.append(video.video(path.dirname(file),path.basename(file)))
            if videosObj[-1].fileBaseName in name_file:
                name_file[videosObj[-1].fileBaseName] += 1
                videosObj[-1].fileBaseName += "_"+str(name_file[videosObj[-1].fileBaseName])
            else:
                name_file[videosObj[-1].fileBaseName] = 0
    else:
        for file in files:
            videosObj.append(video.video(inFolder,file))
            if videosObj[-1].fileBaseName in name_file:
                name_file[videosObj[-1].fileBaseName] += 1
                videosObj[-1].fileBaseName += "_"+str(name_file[videosObj[-1].fileBaseName])
            else:
                name_file[videosObj[-1].fileBaseName] = 0
    name_file = None
    
    mergeRules = tools.config_loader(args.config,"mergerules")
    audioRules = decript_merge_rules(mergeRules['audio'])
    
    for videoObj in videosObj:
        videoObj.get_mediadata()
        if isinstance(videoObj.video, list):
            raise Exception(f"Multiple video in the same file {videoObj.filePath}, I can't compare it")

    commonLanguages = video.get_common_audios_language(videosObj)
    if len(commonLanguages) == 0:
        raise Exception("No common language between "+str([videoObj.filePath for videoObj in videosObj]))
    commonLanguages = list(commonLanguages)
    = get_delay_and_best_video(videosObj,commonLanguages.pop(),audioRules)
    for language in commonLanguages:
        pass
    
    print(f"{baseVideoObj.fileName} {delays}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='This script process mkv,mp4 file to generate best file', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-f", "--file", metavar='file', type=str,
                       required=True, help="File(s) you want merge separate with commat")
    parser.add_argument("--folder", metavar='folder', type=str,
                       default=None, help="If All files are in the same folder, in this option give the path of the folder who contain files to merge and don't write it for all files")
    parser.add_argument("-c","--core", metavar='core', type=int, default=1, help="number of core the software can use")
    parser.add_argument("-o","--out", metavar='outdir', type=str, default=".", help="Folder where send new files")
    parser.add_argument("--tmp", metavar='tmpdir', type=str,
                        default="/tmp", help="Folder where send temporar files")
    parser.add_argument("--config", metavar='configFile', type=str,
                        default="config.ini", help="Path to the config file, by default use the config in the software folder")
    parser.add_argument("--pwd", metavar='pwd', type=str,
                        default=".", help="Path to the software, put it if you use the folder from another folder")
    args = parser.parse_args()
    
    chdir(args.pwd)
    import tools
    tools.software = tools.config_loader(args.config, "software")
    tools.tmpFolder = path.join(args.tmp,"VMSAM_"+str(datetime.now().strftime("%Y-%m-%d_%H:%M:%S")))
    if (not tools.make_dirs(tools.tmpFolder)):
        raise Exception("Impossible to create the temporar dir")
    import video
    from audioCorrelation import correlate, test_calcul_can_be, second_correlation
    if args.core > 1:
        tools.core_to_use = args.core-1
    else:
        tools.core_to_use = args.core
    video.ffmpeg_pool = Pool(processes=1)
    
    mergeVideo(set(args.file.split(",")), args.out, args.folder)