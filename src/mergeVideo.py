'''
Created on 24 Apr 2022

@author: studyfranco

This software need libchromaprint-tools,ffmpeg,mediainfo
'''

import argparse
import sys
import traceback
from datetime import datetime
from multiprocessing import Pool
from os import path,chdir
from random import shuffle
from statistics import variance,mean
from time import strftime,gmtime
from threading import Thread

max_delay_variance_second_method = 0.005
cut_file_to_get_delay_second_method = 2.5 # With the second method we need a better result. After we check the two file is compatible, we need a serious right result adjustment

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
        maxTime = 10
    for videoObj in videosObj:
        videoObj.extract_audio_in_part(language,audioParam,cutTime=[["00:00:00",timeTake]])
        videoObj.wait_end_ffmpeg_progress()
        if (not test_calcul_can_be(videoObj.tmpFiles['audio'][0][0],maxTime)):
            audioParam["Format"] = "AAC"
            audioParam['Channels'] = "2"
            if int(audioParam["BitRate"]) > 256000:
                audioParam["BitRate"] = "256000"
            videoObj.extract_audio_in_part(language,audioParam,cutTime=[["00:00:00",timeTake]])
            videoObj.wait_end_ffmpeg_progress()
            if (not test_calcul_can_be(videoObj.tmpFiles['audio'][0][0],maxTime)):
                audioParam["Format"] = "MP3"
                videoObj.extract_audio_in_part(language,audioParam,cutTime=[["00:00:00",timeTake]])
                videoObj.wait_end_ffmpeg_progress()
                if (not test_calcul_can_be(videoObj.tmpFiles['audio'][0][0],maxTime)):
                    raise Exception(f"Audio parameters to get the fidelity not working with {videoObj.filePath}")

def get_delay_fidelity(video_obj_1,video_obj_2,lenghtTime,ignore_audio_couple=set()):
    delay_Fidelity_Values = {}
    video_obj_1.wait_end_ffmpeg_progress()
    video_obj_2.wait_end_ffmpeg_progress()
    for i in range(0,len(video_obj_1.tmpFiles['audio'])):
        for j in range(0,len(video_obj_2.tmpFiles['audio'])):
            if f"{i}-{j}" not in ignore_audio_couple:
                delay_between_two_audio = []
                delay_Fidelity_Values[f"{i}-{j}"] = delay_between_two_audio
                for h in range(0,video.number_cut):
                    delay_between_two_audio.append(correlate(video_obj_1.tmpFiles['audio'][i][h],video_obj_2.tmpFiles['audio'][j][h],lenghtTime))
    return delay_Fidelity_Values

def get_delay_by_second_method(video_obj_1,video_obj_2,ignore_audio_couple=set()):
    delay_Values = {}
    video_obj_1.wait_end_ffmpeg_progress()
    video_obj_2.wait_end_ffmpeg_progress()
    for i in range(0,len(video_obj_1.tmpFiles['audio'])):
        for j in range(0,len(video_obj_2.tmpFiles['audio'])):
            if f"{i}-{j}" not in ignore_audio_couple:
                delay_between_two_audio = []
                delay_Values[f"{i}-{j}"] = delay_between_two_audio
                for h in range(0,video.number_cut):
                    delay_between_two_audio.append(second_correlation(video_obj_1.tmpFiles['audio'][i][h],video_obj_2.tmpFiles['audio'][j][h]))
    return delay_Values

class compare_video(Thread):
    '''
    classdocs
    '''


    def __init__(self, video_obj_1,video_obj_2,begin_in_second,audioParam,language,lenghtTime,lenghtTimePrepare,list_cut_begin_length,time_by_test_best_quality_converted,process_to_get_best_video=True):
        '''
        Constructor
        '''
        Thread.__init__(self)
        self.video_obj_1 = video_obj_1
        self.video_obj_2 = video_obj_2
        self.begin_in_second = begin_in_second
        self.audioParam = audioParam
        self.language = language
        self.lenghtTime = lenghtTime
        self.lenghtTimePrepare = lenghtTimePrepare
        self.list_cut_begin_length = list_cut_begin_length
        self.time_by_test_best_quality_converted = time_by_test_best_quality_converted
        self.video_obj_with_best_quality = None
        self.process_to_get_best_video = process_to_get_best_video

    def run(self):
        try:
            delay = self.test_if_constant_good_delay()
            if self.process_to_get_best_video:
                self.get_best_video(delay)
            else: # You must have the video you want process in video_obj_1
                self.video_obj_1.extract_audio_in_part(self.language,self.audioParam,cutTime=self.list_cut_begin_length)
                self.video_obj_with_best_quality = self.video_obj_1
                self.video_obj_2.delays[self.language] += (delay*-1.0) # Delay you need to give to mkvmerge to be good.
        except Exception as e:
            sys.stderr.write(str(e)+"\n")
        
    def test_if_constant_good_delay(self):
        try:
            delay_first_method,ignore_audio_couple = self.first_delay_test()
            self.recreate_files_for_delay_adjuster(delay_first_method-500)
            delay_second_method = self.second_delay_test(delay_first_method-500,ignore_audio_couple)
            
            calculated_delay = delay_first_method+round(delay_second_method*1000)-500
            if abs(calculated_delay-delay_first_method) < 500:
                return calculated_delay
            else:
                raise Exception(f"Delay found between {self.video_obj_1.filePath} and {self.video_obj_2.filePath} is unexpected between the two methods")
        except Exception as e:
            self.video_obj_1.extract_audio_in_part(self.language,self.audioParam,cutTime=self.list_cut_begin_length)
            self.video_obj_2.extract_audio_in_part(self.language,self.audioParam,cutTime=self.list_cut_begin_length)
            raise e
        
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
            raise Exception(f"Multiple delay found with the method 1 and in test 1 for {self.video_obj_1.filePath} and {self.video_obj_2.filePath}")
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
            elif delay_fidelity_list[0][2] ==  delay_fidelity_list[-1][2]:
                sys.stderr.write(f"Multiple delay found with the method 1 and in test 2 for {self.video_obj_1.filePath} and {self.video_obj_2.filePath} but the first and last part have the same delay\n")
                delay_detected.add(delay_fidelity_list[0][2])
            else:
                raise Exception(f"Multiple delay found with the method 1 and in test 2 for {self.video_obj_1.filePath} and {self.video_obj_2.filePath}")
                    
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
                elif delay_fidelity_list[0][2] ==  delay_fidelity_list[-1][2]:
                    sys.stderr.write(f"Multiple delay found with the method 1 and in test 3 for {self.video_obj_1.filePath} and {self.video_obj_2.filePath} but the first and last part have the same delay\n")
                    delay_detected.add(delay_fidelity_list[0][2])
                else:
                    raise Exception(f"Multiple delay found with the method 1 and in test 3 for {self.video_obj_1.filePath} and {self.video_obj_2.filePath}")
                        
            if len(delay_detected) == 1 and 0 in delay_detected:
                return delayUse,ignore_audio_couple
            else:
                raise Exception(f"Not able to find delay with the method 1 and in test 4 for {self.video_obj_1} and {self.video_obj_2}")
        
    def recreate_files_for_delay_adjuster(self,delay_use):
        list_cut_begin_length = video.generate_cut_with_begin_length(self.begin_in_second+(delay_use/1000),self.lenghtTime,self.lenghtTimePrepare)
        self.video_obj_2.extract_audio_in_part(self.language,self.audioParam,cutTime=list_cut_begin_length)
        
    def second_delay_test(self,delayUse,ignore_audio_couple):
        global max_delay_variance_second_method
        global cut_file_to_get_delay_second_method
        delay_Values = get_delay_by_second_method(self.video_obj_1,self.video_obj_2,ignore_audio_couple=ignore_audio_couple)
        delay_detected = set()
        for key_audio, delay_list in delay_Values.items():
            set_delay = set()
            for delay in delay_list:
                set_delay.add(delay[1])
            if variance(set_delay) < max_delay_variance_second_method:
                delay_detected.update(set_delay)
            elif (delay_list[0][1]-delay_list[-1][1]) < max_delay_variance_second_method:
                sys.stderr.write(f"Variance delay in the second test is to big with {self.video_obj_1.filePath} and {self.video_obj_2.filePath} ")
                delay_detected.add(delay_list[0][1])
                delay_detected.add(delay_list[-1][1])
            else:
                raise Exception(f"Variance delay in the second test is to big with {self.video_obj_1.filePath} and {self.video_obj_2.filePath} but the first and last part have the similar delay\n")
        
        if variance(delay_detected) > max_delay_variance_second_method:
            raise Exception(f"Multiple delay found with the method 2 and in test 1 for {self.video_obj_1.filePath} and {self.video_obj_2.filePath} at the second method")
        else:
            self.video_obj_1.extract_audio_in_part(self.language,self.audioParam,cutTime=[[strftime('%H:%M:%S',gmtime(int(self.begin_in_second))),strftime('%H:%M:%S',gmtime(int(self.lenghtTime*video.number_cut/cut_file_to_get_delay_second_method)))]])
            begining_in_second, begining_in_millisecond = video.get_begin_time_with_millisecond(delayUse,self.begin_in_second)
            self.video_obj_2.extract_audio_in_part(self.language,self.audioParam,cutTime=[[strftime('%H:%M:%S',gmtime(begining_in_second))+begining_in_millisecond,strftime('%H:%M:%S',gmtime(int(self.lenghtTime*video.number_cut/cut_file_to_get_delay_second_method)))]])
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
            
    def get_best_video(self,delay):
        delay,begins_video_for_compare_quality = video.get_good_frame(self.video_obj_1, self.video_obj_2, self.begin_in_second, self.lenghtTime, self.time_by_test_best_quality_converted, (delay/1000))

        if video.get_best_quality_video(self.video_obj_1, self.video_obj_2, begins_video_for_compare_quality, self.time_by_test_best_quality_converted) == 1:
            self.video_obj_1.extract_audio_in_part(self.language,self.audioParam,cutTime=self.list_cut_begin_length)
            self.video_obj_with_best_quality = self.video_obj_1
            self.video_obj_2.delays[self.language] += (delay*-1.0) # Delay you need to give to mkvmerge to be good.
        else:
            self.video_obj_2.extract_audio_in_part(self.language,self.audioParam,cutTime=self.list_cut_begin_length)
            self.video_obj_with_best_quality = self.video_obj_2
            self.video_obj_1.delays[self.language] += delay # Delay you need to give to mkvmerge to be good.

def was_they_not_already_compared(video_obj_1,video_obj_2,already_compared):
    name_in_list = [video_obj_1.filePath,video_obj_2.filePath]
    sorted(name_in_list)
    return (name_in_list[0] not in already_compared or (name_in_list[0] in already_compared and name_in_list[1] not in already_compared[name_in_list[0]]))

def can_always_compare_it(video_obj,compare_objs,new_compare_objs,already_compared):
    for other_video_obj in compare_objs:
        if was_they_not_already_compared(video_obj,other_video_obj,already_compared):
            return True
    for other_video_obj in new_compare_objs:
        if was_they_not_already_compared(video_obj,other_video_obj,already_compared):
            return True
    return False

def get_waiter_to_compare(video_obj,new_compare_objs,already_compared):
    for i in range(0,len(new_compare_objs),1):
        if was_they_not_already_compared(video_obj,new_compare_objs[i],already_compared):
            return new_compare_objs.pop(i)
    return None

"""
    Theorically the video I will remove have no connexion between other files.
"""
def remove_not_compatible_audio(video_obj_path_file,already_compared):
    other_videos_path_file = []
    if video_obj_path_file in already_compared:
        for other_video_path_file, is_the_best_video in already_compared[video_obj_path_file].items():
            if is_the_best_video != None:
                if is_the_best_video:
                    other_videos_path_file.append(other_video_path_file)
                else:
                    raise Exception(f"You know what ? In remove_not_compatible_audio we don't wait a result 'False' here. What happen with {video_obj_path_file} and {other_video_path_file} ?")
        del already_compared[video_obj_path_file]
    
    for other_video_path_file, dict_with_results in already_compared.items():
        if video_obj_path_file in dict_with_results:
            if dict_with_results[video_obj_path_file] == None:
                del dict_with_results[video_obj_path_file]
            elif dict_with_results[video_obj_path_file]:
                raise Exception(f"You know what ? In remove_not_compatible_audio we don't wait a result 'True' here. What happen with {video_obj_path_file} and {other_video_path_file} ?")
            else:
                other_videos_path_file.append(other_video_path_file)
                del dict_with_results[video_obj_path_file]
    
    for other_video_path_file in other_videos_path_file:
        other_videos_path_file.extend(remove_not_compatible_audio(other_video_path_file,already_compared))
    
    return other_videos_path_file

def prepare_get_delay(videosObj,language,audioRules):
    worseAudioQualityWillUse = video.get_worse_quality_audio_param(videosObj,language,audioRules)
    min_video_duration_in_sec = video.get_shortest_audio_durations(videosObj,language)
    get_good_parameters_to_get_fidelity(videosObj,language,worseAudioQualityWillUse,min_video_duration_in_sec)
    
    begin_in_second,length_time = video.generate_begin_and_length_by_segment(min_video_duration_in_sec)
    length_time_converted = strftime('%H:%M:%S',gmtime(length_time))
    list_cut_begin_length = video.generate_cut_with_begin_length(begin_in_second,length_time,length_time_converted)
    
    for videoObj in videosObj:
        videoObj.extract_audio_in_part(language,worseAudioQualityWillUse,cutTime=list_cut_begin_length)
        videoObj.delays[language] = 0
        for language,audios in videoObj.audios.items():
            for audio in audios:
                audio["keep"] = True
        for language,audios in videoObj.commentary.items():
            for audio in audios:
                audio["keep"] = (not special_params["remove_commentary"])
    
    return begin_in_second,worseAudioQualityWillUse,length_time,length_time_converted,list_cut_begin_length

def print_forced_video(forced_best_video):
    if tools.dev:
        print(f"The forced video is {forced_best_video}")

def remove_not_compatible_video(list_not_compatible_video,dict_file_path_obj):
    if len(list_not_compatible_video):
        from sys import stderr
        stderr.write(f"{[not_compatible_video for not_compatible_video in list_not_compatible_video]} not compatible with the others videos")
        stderr.write("\n")
        for not_compatible_video in list_not_compatible_video:
            if not_compatible_video in dict_file_path_obj:
                del dict_file_path_obj[not_compatible_video]
        if len(dict_file_path_obj) < 2:
            raise Exception(f"Only {dict_file_path_obj.keys()} file left. This is useless to merge files")
    
def get_delay_and_best_video(videosObj,language,audioRules,dict_file_path_obj):
    begin_in_second,worseAudioQualityWillUse,length_time,length_time_converted,list_cut_begin_length = prepare_get_delay(videosObj,language,audioRules)
    
    time_by_test_best_quality_converted = strftime('%H:%M:%S',gmtime(video.generate_time_compare_video_quality(length_time)))
    
    compareObjs = videosObj.copy()
    already_compared = {}
    list_not_compatible_video = []
    while len(compareObjs) > 1:
        if len(compareObjs)%2 != 0:
            new_compare_objs = [compareObjs.pop()]
        else:
            new_compare_objs = []
        list_in_compare_video = []
        for i in range(0,len(compareObjs),2):
            if was_they_not_already_compared(compareObjs[i],compareObjs[i+1],already_compared):
                list_in_compare_video.append(compare_video(compareObjs[i],compareObjs[i+1],begin_in_second,worseAudioQualityWillUse,language,length_time,length_time_converted,list_cut_begin_length,time_by_test_best_quality_converted))
                list_in_compare_video[-1].start()
            elif len(new_compare_objs):
                compare_new_obj = None
                remove_i = False
                remove_i_1 = False
                if can_always_compare_it(compareObjs[i],compareObjs,new_compare_objs,already_compared):
                    compare_new_obj = get_waiter_to_compare(compareObjs[i],new_compare_objs,already_compared)
                    if compare_new_obj != None:
                        list_in_compare_video.append(compare_video(compareObjs[i],compare_new_obj,begin_in_second,worseAudioQualityWillUse,language,length_time,length_time_converted,list_cut_begin_length,time_by_test_best_quality_converted))
                        list_in_compare_video[-1].start()
                    else:
                        new_compare_objs.append(compareObjs[i])
                else:
                    remove_i = True
                if can_always_compare_it(compareObjs[i+1],compareObjs,new_compare_objs,already_compared):
                    compare_new_obj = get_waiter_to_compare(compareObjs[i+1],new_compare_objs,already_compared)
                    if compare_new_obj != None:
                        list_in_compare_video.append(compare_video(compareObjs[i+1],compare_new_obj,begin_in_second,worseAudioQualityWillUse,language,length_time,length_time_converted,list_cut_begin_length,time_by_test_best_quality_converted))
                        list_in_compare_video[-1].start()
                    else:
                        new_compare_objs.append(compareObjs[i+1])
                elif compare_new_obj != None and was_they_not_already_compared(compareObjs[i+1],compare_new_obj,already_compared):
                    new_compare_objs.append(compareObjs[i+1])
                else:
                    remove_i_1 = True

                if remove_i and remove_i_1:
                    """
                        TODO:
                            I want check the file with the best number of match file and remove the files with the less connected. Normally the two list can't be connected.
                    """
                    remove_i = False
                if remove_i:
                    list_not_compatible_video.append(compareObjs[i].filePath)
                    list_not_compatible_video.extend(remove_not_compatible_audio(compareObjs[i].filePath,already_compared))
                elif remove_i_1:
                    list_not_compatible_video.append(compareObjs[i+1].filePath)
                    list_not_compatible_video.extend(remove_not_compatible_audio(compareObjs[i+1].filePath,already_compared))
            else:
                """
                    TODO:
                        I want check the file with the best number of match file and remove the files with the less connected. Normally the two list can't be connected.
                """
                from sys import stderr
                stderr.write(f"You enter in a not working part. You have one last file not compatible you may stop here the result will be random")
                stderr.write("\n")
                list_not_compatible_video.append(compareObjs[i+1].filePath)
                list_not_compatible_video.extend(remove_not_compatible_audio(compareObjs[i+1].filePath,already_compared))
        
        compareObjs = new_compare_objs
        for compare_video_obj in list_in_compare_video:
            nameInList = [compare_video_obj.video_obj_1.filePath,compare_video_obj.video_obj_2.filePath]
            sorted(nameInList)
            compare_video_obj.join()
            if compare_video_obj.video_obj_with_best_quality != None:
                is_the_best_video = compare_video_obj.video_obj_with_best_quality.filePath==nameInList[0]
                compareObjs.append(compare_video_obj.video_obj_with_best_quality)
            else:
                is_the_best_video = None
                compareObjs.append(compare_video_obj.video_obj_1)
                compareObjs.append(compare_video_obj.video_obj_2)
            if nameInList[0] in already_compared:
                already_compared[nameInList[0]][nameInList[1]] = is_the_best_video
            else:
                already_compared[nameInList[0]] = {nameInList[1]: is_the_best_video}
            
        shuffle(compareObjs)
    
    remove_not_compatible_video(list_not_compatible_video,dict_file_path_obj)
    return already_compared

def get_delay(videosObj,language,audioRules,dict_file_path_obj,forced_best_video):
    begin_in_second,worseAudioQualityWillUse,length_time,length_time_converted,list_cut_begin_length = prepare_get_delay(videosObj,language,audioRules)
    
    videosObj.remove(dict_file_path_obj[forced_best_video])
    launched_compare = compare_video(dict_file_path_obj[forced_best_video],videosObj[0],begin_in_second,worseAudioQualityWillUse,language,length_time,length_time_converted,list_cut_begin_length,0,process_to_get_best_video=False)
    launched_compare.start()
    
    already_compared = {forced_best_video:{}}
    list_not_compatible_video = []
    for i in range(1,len(videosObj)):
        prepared_compare = compare_video(dict_file_path_obj[forced_best_video],videosObj[i],begin_in_second,worseAudioQualityWillUse,language,length_time,length_time_converted,list_cut_begin_length,0,process_to_get_best_video=False)
        launched_compare.join()
        prepared_compare.start()
        if launched_compare.video_obj_with_best_quality != None:
            already_compared[forced_best_video][launched_compare.video_obj_2.filePath] = True
        else:
            list_not_compatible_video.append(launched_compare.video_obj_2.filePath)
        launched_compare = prepared_compare
    
    videosObj.append(dict_file_path_obj[forced_best_video])
    launched_compare.join()
    if launched_compare.video_obj_with_best_quality != None:
        already_compared[forced_best_video][launched_compare.video_obj_2.filePath] = True
    else:
        list_not_compatible_video.append(launched_compare.video_obj_2.filePath)
    
    remove_not_compatible_video(list_not_compatible_video,dict_file_path_obj)
    return already_compared

def generate_merge_command_insert_ID_audio_track_to_remove_and_new_und_language(merge_cmd,video_audio_track_list,video_commentary_track_list):
    if len(video_audio_track_list) == 2 and "und" in video_audio_track_list and tools.default_language_for_undetermine != "und":
        # This step is linked by the fact if you have und audio they are orginialy convert in another language
        # This was convert in a language, but the object is the same and can be compared
        if video_audio_track_list[tools.default_language_for_undetermine] == video_audio_track_list['und']:
            del video_audio_track_list[tools.default_language_for_undetermine]
        
    track_to_remove = set()
    for language,audios in video_audio_track_list.items():
        for audio in audios:
            if (not audio["keep"]):
                track_to_remove.add(audio["StreamOrder"])
            elif language == "und" and special_params["change_all_und"]:
                merge_cmd.extend(["--language", audio["StreamOrder"]+":"+tools.default_language_for_undetermine])
                if tools.default_language_for_undetermine == special_params["original_language"]:
                    merge_cmd.extend(["--original-flag", audio["StreamOrder"]])
            elif language == special_params["original_language"]:
                merge_cmd.extend(["--original-flag", audio["StreamOrder"]])
            merge_cmd.extend(["--forced-display-flag", audio["StreamOrder"]+":0", "--default-track-flag", audio["StreamOrder"]+":0"])
    for language,audios in video_commentary_track_list.items():
        for audio in audios:
            if (not audio["keep"]):
                track_to_remove.add(audio["StreamOrder"])
            else:
                if language == "und" and special_params["change_all_und"]:
                    merge_cmd.extend(["--language", audio["StreamOrder"]+":"+tools.default_language_for_undetermine])
                merge_cmd.extend(["--commentary-flag", audio["StreamOrder"], "--forced-display-flag", audio["StreamOrder"]+":0", "--default-track-flag", audio["StreamOrder"]+":0"])

    if len(track_to_remove):
        merge_cmd.extend(["-a"],"!"+",".join(track_to_remove))

def generate_merge_command_other_part(video_path_file,dict_list_video_win,dict_file_path_obj,merge_cmd,delay_winner,common_language_use_for_generate_delay):
    video_obj = dict_file_path_obj[video_path_file]
    delay_to_put = video_obj.delays[common_language_use_for_generate_delay] + delay_winner
    generate_merge_command_insert_ID_audio_track_to_remove_and_new_und_language(merge_cmd,video_obj.audios,video_obj.commentary)
    if delay_to_put != 0:
        merge_cmd.extend(["--sync", f"-1:{int(delay_to_put)}"])
    merge_cmd.extend(["-D", video_obj.filePath])
    
    print(f'\t{video_obj.filePath} will add with a delay of {int(delay_to_put)}')
    
    if video_path_file in dict_list_video_win:
        for other_video_path_file in dict_list_video_win[video_path_file]:
            generate_merge_command_other_part(other_video_path_file,dict_list_video_win,dict_file_path_obj,merge_cmd,delay_to_put,common_language_use_for_generate_delay)

def generate_launch_merge_command(dict_with_video_quality_logic,dict_file_path_obj,out_folder,common_language_use_for_generate_delay):
    set_bad_video = set()
    dict_list_video_win = {}
    for video_path_file, dict_with_results in dict_with_video_quality_logic.items():
        for other_video_path_file, is_the_best_video in  dict_with_results.items():
            if is_the_best_video:
                set_bad_video.add(other_video_path_file)
                if video_path_file in dict_list_video_win:
                    dict_list_video_win[video_path_file].append(other_video_path_file)
                else:
                    dict_list_video_win[video_path_file] = [other_video_path_file]
            else:
                set_bad_video.add(video_path_file)
                if other_video_path_file in dict_list_video_win:
                    dict_list_video_win[other_video_path_file].append(video_path_file)
                else:
                    dict_list_video_win[other_video_path_file] = [video_path_file]
    
    best_video = dict_file_path_obj[list(dict_file_path_obj.keys() - set_bad_video)[0]]
    print(f'The best video path is {best_video.filePath}')
    out_path_file_name = path.join(out_folder,f"{best_video.fileBaseName}_merged")
    if path.exists(out_path_file_name+'.mkv'):
        i = 1
        while path.exists(out_path_file_name+f'_({str(i)}).mkv'):
            i =+ 1
        out_path_file_name += f'_({str(i)}).mkv'
    else:
        out_path_file_name += '.mkv'
    merge_cmd = [tools.software["mkvmerge"], "-o", out_path_file_name]
    generate_merge_command_insert_ID_audio_track_to_remove_and_new_und_language(merge_cmd,best_video.audios,best_video.commentary)
    if special_params["change_all_und"] and 'Language' not in best_video.video:
        merge_cmd.extend(["--language", best_video.video["StreamOrder"]+":"+tools.default_language_for_undetermine])
    merge_cmd.append(best_video.filePath)
    for other_video_path_file in dict_list_video_win[best_video.filePath]:
        generate_merge_command_other_part(other_video_path_file,dict_list_video_win,dict_file_path_obj,merge_cmd,best_video.delays[common_language_use_for_generate_delay],common_language_use_for_generate_delay)
    
    print(" ".join(merge_cmd))
    tools.launch_cmdExt(merge_cmd)
    
def simple_merge_video(videosObj,audioRules,out_folder,dict_file_path_obj,forced_best_video):
    if forced_best_video == None:
        min_video_duration_in_sec = video.get_shortest_video_durations(videosObj)
        begin_in_second,length_time = video.generate_begin_and_length_by_segment(min_video_duration_in_sec)
        time_by_test_best_quality_converted = strftime('%H:%M:%S',gmtime(video.generate_time_compare_video_quality(length_time)))
        begins_video_for_compare_quality = video.generate_cut_to_compare_video_quality(begin_in_second,begin_in_second,length_time)

        compareObjs = videosObj.copy()
        dict_with_video_quality_logic = {}
        while len(compareObjs) > 1:
            if len(compareObjs)%2 != 0:
                new_compare_objs = [compareObjs.pop()]
            else:
                new_compare_objs = []
            for i in range(0,len(compareObjs),2):
                nameInList = [compareObjs[i].filePath,compareObjs[i+1].filePath]
                sorted(nameInList)
                if video.get_best_quality_video(dict_file_path_obj[nameInList[0]], dict_file_path_obj[nameInList[1]], begins_video_for_compare_quality, time_by_test_best_quality_converted) == 1:
                    is_the_best_video = True
                    new_compare_objs.append(dict_file_path_obj[nameInList[0]])
                else:
                    is_the_best_video = False
                    new_compare_objs.append(dict_file_path_obj[nameInList[1]])
                if nameInList[0] in dict_with_video_quality_logic:
                    dict_with_video_quality_logic[nameInList[0]][nameInList[1]] = is_the_best_video
                else:
                    dict_with_video_quality_logic[nameInList[0]] = {nameInList[1]: is_the_best_video}
            compareObjs = new_compare_objs
    else:
        print_forced_video(forced_best_video)
        dict_with_video_quality_logic = {forced_best_video:{}}
        for file_path in dict_file_path_obj:
            if forced_best_video != file_path:
                dict_with_video_quality_logic[forced_best_video] = {file_path:True}
    
    for videoObj in videosObj:
        videoObj.delays["und"] = 0
        for language,audios in videoObj.audios.items():
            for audio in audios:
                audio["keep"] = True
        for language,audios in videoObj.commentary.items():
            for audio in audios:
                audio["keep"] = (not special_params["remove_commentary"])
        
    generate_launch_merge_command(dict_with_video_quality_logic,dict_file_path_obj,out_folder,"und")
    
def sync_merge_video(videosObj,audioRules,out_folder,dict_file_path_obj,forced_best_video):
    commonLanguages = video.get_common_audios_language(videosObj)
    try:
        commonLanguages.remove("und")
    except:
        pass
    if len(commonLanguages) == 0:
        raise Exception("No common language between "+str([videoObj.filePath for videoObj in videosObj]))
    commonLanguages = list(commonLanguages)
    common_language_use_for_generate_delay = commonLanguages.pop()
    if forced_best_video == None:
        dict_with_video_quality_logic = get_delay_and_best_video(videosObj,common_language_use_for_generate_delay,audioRules,dict_file_path_obj)
    else:
        print_forced_video(forced_best_video)
        dict_with_video_quality_logic = get_delay(videosObj,common_language_use_for_generate_delay,audioRules,dict_file_path_obj,forced_best_video)
    for language in commonLanguages:
        """
        TODO:
            This part will use for a new audio correlation. They will only be use to cross validate the correlation and detect some big errors.
        """
        pass
    
    generate_launch_merge_command(dict_with_video_quality_logic,dict_file_path_obj,out_folder,common_language_use_for_generate_delay)
    
def merge_videos(files,out_folder,merge_sync,inFolder=None):
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
    
    dict_file_path_obj = {}
    forced_best_video = None
    for videoObj in videosObj:
        process_mediadata_thread = Thread(target=videoObj.get_mediadata)
        process_mediadata_thread.start()
        dict_file_path_obj[videoObj.filePath] = videoObj
        if special_params["forced_best_video"] != "":
            if special_params["forced_best_video_contain"]:
                if special_params["forced_best_video"] in videoObj.fileName:
                    forced_best_video = videoObj.filePath
            elif videoObj.fileName  == special_params["forced_best_video"] or videoObj.filePath == special_params["forced_best_video"]:
                forced_best_video = videoObj.filePath
        process_mediadata_thread.join()
    
    if merge_sync:
        sync_merge_video(videosObj,audioRules,out_folder,dict_file_path_obj,forced_best_video)
    else:
        simple_merge_video(videosObj,audioRules,out_folder,dict_file_path_obj,forced_best_video)
    
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
                        default="config.ini", help="Path to the config file, by default use the config in the software folder. This config is for configure the path to your softwares")
    parser.add_argument("--param", metavar='param', type=str,
                       default=None, help="Give the path to a special file for your merge.")
    parser.add_argument("--pwd", metavar='pwd', type=str,
                        default=".", help="Path to the software, put it if you use the folder from another folder")
    parser.add_argument("--noSync", dest='noSync', default=False, action='store_true', help="If you don't want research a audio sync between files")
    parser.add_argument("--dev", dest='dev', default=False, action='store_true', help="Print more errors and write all logs")
    args = parser.parse_args()
    
    chdir(args.pwd)
    import tools
    tools.tmpFolder = path.join(args.tmp,"VMSAM_"+str(datetime.now().strftime("%Y-%m-%d_%H:%M:%S")))
    import video
    from audioCorrelation import correlate, test_calcul_can_be, second_correlation
    if args.core > 1:
        tools.core_to_use = args.core-1
    else:
        tools.core_to_use = args.core
    video.ffmpeg_pool = Pool(processes=2)
    
    tools.dev = args.dev
    
    try:
        tools.software = tools.config_loader(args.config, "software")
        if (not tools.make_dirs(tools.tmpFolder)):
            raise Exception("Impossible to create the temporar dir")
        if args.param != None:
            import json
            with open(args.param) as param_file:
                special_params = json.load(param_file)
            tools.default_language_for_undetermine = special_params["default_language_und"]
            if "model_path" in special_params and special_params['model_path'] != "" and special_params['model_path'] != None:
                video.path_to_livmaf_model = ":model_path="+special_params['model_path']
        else:
            special_params = {"change_all_und":False, "original_language":"", "remove_commentary":False, "forced_best_video":"", "forced_best_video_contain":False}
        merge_videos(set(args.file.split(",")), args.out, (not args.noSync), args.folder)
        tools.remove_dir(tools.tmpFolder)
    except:
        tools.remove_dir(tools.tmpFolder)
        traceback.print_exc()
        exit(1)
    exit(0)