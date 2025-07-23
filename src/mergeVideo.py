'''
Created on 24 Apr 2022

@author: studyfranco

This software need libchromaprint-tools,ffmpeg,mediainfo
'''

import re
import sys
import traceback
from os import path
from random import shuffle
from statistics import variance,mean
from time import strftime,gmtime
from threading import Thread,RLock
import tools
import video
from audioCorrelation import correlate, test_calcul_can_be, second_correlation
import json
from decimal import *

max_delay_variance_second_method = 0.005
cut_file_to_get_delay_second_method = 2.5 # With the second method we need a better result. After we check the two file is compatible, we need a serious right result adjustment

errors_merge = []
errors_merge_lock = RLock()
max_stream = 85

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
                            rules[subValue][sup[0]] = False
                elif isinstance(sup[1], bool):
                    for subValue in value:
                        rules[sup[0]][subValue] = True
                        rules[subValue][sup[0]] = False
                    
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
        videoObj.wait_end_ffmpeg_progress_audio()
        if (not test_calcul_can_be(videoObj.tmpFiles['audio'][0][0],maxTime)):
            raise Exception(f"Audio parameters to get the fidelity not working with {videoObj.filePath}")

def get_delay_fidelity(video_obj_1,video_obj_2,lenghtTime,ignore_audio_couple=set()):
    delay_Fidelity_Values = {}
    video_obj_1.wait_end_ffmpeg_progress_audio()
    video_obj_2.wait_end_ffmpeg_progress_audio()
    for i in range(0,len(video_obj_1.tmpFiles['audio'])):
        for j in range(0,len(video_obj_2.tmpFiles['audio'])):
            if f"{i}-{j}" not in ignore_audio_couple:
                delay_between_two_audio = []
                delay_Fidelity_Values[f"{i}-{j}"] = delay_between_two_audio
                for h in range(0,video.number_cut):
                    delay_between_two_audio.append(correlate(video_obj_1.tmpFiles['audio'][i][h],video_obj_2.tmpFiles['audio'][j][h],lenghtTime))
    import gc
    gc.collect()
    return delay_Fidelity_Values

def get_delay_by_second_method(video_obj_1,video_obj_2,ignore_audio_couple=set()):
    delay_Values = {}
    video_obj_1.wait_end_ffmpeg_progress_audio()
    video_obj_2.wait_end_ffmpeg_progress_audio()
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
        self.uncompatibleaudiofind = set()

    def run(self):
        try:
            delay = self.test_if_constant_good_delay()
            if self.process_to_get_best_video:
                self.get_best_video(delay)
            else: # You must have the video you want process in video_obj_1
                self.video_obj_1.extract_audio_in_part(self.language,self.audioParam,cutTime=self.list_cut_begin_length,asDefault=True)
                self.video_obj_2.remove_tmp_files(type_file="audio")
                self.video_obj_with_best_quality = self.video_obj_1
                delay = self.adjust_delay_to_frame(delay)
                self.video_obj_2.delays[self.language] += (delay*-Decimal(1.0)) # Delay you need to give to mkvmerge to be good.
        except Exception as e:
            traceback.print_exc()
            sys.stderr.write(str(e)+"\n")
            with errors_merge_lock:
                errors_merge.append(str(e))
        
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
            self.video_obj_1.extract_audio_in_part(self.language,self.audioParam,cutTime=self.list_cut_begin_length,asDefault=True)
            self.video_obj_2.extract_audio_in_part(self.language,self.audioParam,cutTime=self.list_cut_begin_length,asDefault=True)
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
            elif delay_fidelity_list[0][2] ==  delay_fidelity_list[-1][2]:
                number_values_not_good = 0
                for delay_fidelity in delay_fidelity_list:
                    if delay_fidelity != delay_fidelity_list[0][2]:
                        number_values_not_good += 1
                    if (number_values_not_good/video.number_cut) > 0.25:
                        ignore_audio_couple.add(key_audio)
                    else:
                        delay_detected.add(delay_fidelity_list[0][2])
            elif len(set_delay) == 2 and abs(list(set_delay)[0]-list(set_delay)[1]) < 127:
                to_ignore = set(delay_Fidelity_Values.keys())
                to_ignore.remove(key_audio)
                set_delay_clone = list(set_delay.copy())
                delay_found = self.adjuster_chroma_bugged(list(set_delay),to_ignore)
                if delay_found == None:
                    ignore_audio_couple.add(key_audio)
                else:
                    #delay_detected.add(delay_fidelity_list[0][2])
                    if set_delay_clone[1] > set_delay_clone[0]:
                        delay_detected.add(set_delay_clone[0]+67) # 125/2
                    else:
                        delay_detected.add(set_delay_clone[1]+67)
            else:
                # Work in progress
                # We need to ask to the user to pass them if they want.
                ignore_audio_couple.add(key_audio)
        
        '''
            TODO:
                Detect if the audio is always not compatible. Set it not compatible in it. (And not convert it all the time)
        '''
        if len(delay_detected) != 1:
            delayUse = None
            if len(delay_detected) == 2 and abs(list(delay_detected)[0]-list(delay_detected)[1]) < 127:
                delayUse = self.adjuster_chroma_bugged(list(delay_detected),ignore_audio_couple)
            if delayUse == None:
                delays = self.get_delays_dict(delay_Fidelity_Values,0)
                self.video_obj_1.delayFirstMethodAbort[self.video_obj_2.filePath] = [1,delays]
                self.video_obj_2.delayFirstMethodAbort[self.video_obj_1.filePath] = [2,delays]
                raise Exception(f"Multiple delay found with the method 1 and in test 1 {delay_Fidelity_Values} for {self.video_obj_1.filePath} and {self.video_obj_2.filePath}")
            else:
                sys.stderr.write(f"This is  delay {delayUse}, calculated by second method for {self.video_obj_1.filePath} and {self.video_obj_2.filePath} \n")
                with errors_merge_lock:
                    errors_merge.append(f"This is  delay {delayUse}, calculated by second method for {self.video_obj_1.filePath} and {self.video_obj_2.filePath} \n")
        elif 'delay_found' in locals() and delay_found != None:
            delayUse = delay_found
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
                sys.stderr.write(f"Multiple delay found with the method 1 and in test 2 {delay_Fidelity_Values} with a delay of {delayUse} for {self.video_obj_1.filePath} and {self.video_obj_2.filePath} but the first and last part have the same delay\n")
                with errors_merge_lock:
                    errors_merge.append(f"Multiple delay found with the method 1 and in test 2 {delay_Fidelity_Values} with a delay of {delayUse} for {self.video_obj_1.filePath} and {self.video_obj_2.filePath} but the first and last part have the same delay\n")
                delay_detected.add(delay_fidelity_list[0][2])
            else:
                number_of_change = 0
                previous_delay = delay_fidelity_list[0][2]
                previous_delay_iteration = 0
                majoritar_value = delay_fidelity_list[0][2]
                majoritar_value_number_iteration = 0
                for delay_data in delay_fidelity_list:
                    if delay_data[2] != previous_delay:
                        number_of_change += 1
                        if majoritar_value_number_iteration < previous_delay_iteration:
                            majoritar_value = previous_delay
                            majoritar_value_number_iteration = previous_delay_iteration
                        previous_delay = delay_data[2]
                        previous_delay_iteration = 1
                    else:
                        previous_delay_iteration += 1
                
                if majoritar_value_number_iteration < previous_delay_iteration:
                    majoritar_value = previous_delay
                    majoritar_value_number_iteration = previous_delay_iteration
                
                if len(set_delay) > 2 or number_of_change > 1:
                    delays = self.get_delays_dict(delay_Fidelity_Values,delayUse)
                    self.video_obj_1.delayFirstMethodAbort[self.video_obj_2.filePath] = [1,delays]
                    self.video_obj_2.delayFirstMethodAbort[self.video_obj_1.filePath] = [2,delays]
                    raise Exception(f"Multiple delay found with the method 1 and in test 2 {delay_Fidelity_Values} with a delay of {delayUse} for {self.video_obj_1.filePath} and {self.video_obj_2.filePath}")
                else:
                    sys.stderr.write(f"Multiple delay found with the method 1 and in test 2 {delay_Fidelity_Values} with a delay of {delayUse} for {self.video_obj_1.filePath} and {self.video_obj_2.filePath} but only one piece have the problem, this is maybe a bug.\n")
                    with errors_merge_lock:
                        errors_merge.append(f"Multiple delay found with the method 1 and in test 2 {delay_Fidelity_Values} with a delay of {delayUse} for {self.video_obj_1.filePath} and {self.video_obj_2.filePath} but only one piece have the problem, this is maybe a bug.\n")
                    delay_detected.add(majoritar_value)
                """delay_adjusted = None
                if len(set_delay) == 2 and abs(list(set_delay)[0]-list(set_delay)[1]) < 127:
                    delay_adjusted = self.adjuster_chroma_bugged(list(set([delayUse + delay_fidelity[2] for delay_fidelity in delay_fidelity_list])),ignore_audio_couple)
                if delay_adjusted == None:
                    
                else:
                    delay_detected.add(delay_adjusted-delayUse)"""
                    
        if len(delay_detected) == 1 and 0 in delay_detected:
            return delayUse,ignore_audio_couple
        elif len(delay_detected) == 0:
            raise Exception("We don't have any delay. Why this happen ?")
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
                    sys.stderr.write(f"Multiple delay found with the method 1 and in test 3 {delay_Fidelity_Values} with a delay of {delayUse} for {self.video_obj_1.filePath} and {self.video_obj_2.filePath} but the first and last part have the same delay\n")
                    with errors_merge_lock:
                        errors_merge.append(f"Multiple delay found with the method 1 and in test 3 {delay_Fidelity_Values} with a delay of {delayUse} for {self.video_obj_1.filePath} and {self.video_obj_2.filePath} but the first and last part have the same delay\n")
                    delay_detected.add(delay_fidelity_list[0][2])
                else:
                    delays = self.get_delays_dict(delay_Fidelity_Values,delayUse=0)
                    self.video_obj_1.delayFirstMethodAbort[self.video_obj_2.filePath] = [1,delays]
                    self.video_obj_2.delayFirstMethodAbort[self.video_obj_1.filePath] = [2,delays]
                    raise Exception(f"Multiple delay found with the method 1 and in test 3 {delay_Fidelity_Values} with a delay of {delayUse} for {self.video_obj_1.filePath} and {self.video_obj_2.filePath}")
                        
            if len(delay_detected) == 1 and 0 in delay_detected:
                return delayUse,ignore_audio_couple
            else:
                delays = self.get_delays_dict(delay_Fidelity_Values,delayUse=0)
                self.video_obj_1.delayFirstMethodAbort[self.video_obj_2.filePath] = [1,delays]
                self.video_obj_2.delayFirstMethodAbort[self.video_obj_1.filePath] = [2,delays]
                raise Exception(f"Not able to find delay with the method 1 and in test 4 we find {delay_detected} with a delay of {delayUse} for {self.video_obj_1.filePath} and {self.video_obj_2.filePath}")
    
    def adjuster_chroma_bugged(self,list_delay,ignore_audio_couple):
        if list_delay[0] > list_delay[1]:
            delay_first_method_lower_result = list_delay[1]
            delay_first_method_bigger_result = list_delay[0]
        else:
            delay_first_method_lower_result = list_delay[0]
            delay_first_method_bigger_result = list_delay[1]
        #self.recreate_files_for_delay_adjuster(delay_first_method_lower_result)
        mean_between_delay = round((list_delay[0]+list_delay[1])/2)
        self.recreate_files_for_delay_adjuster(mean_between_delay)
        try:
            #delay_second_method = self.second_delay_test(delay_first_method_lower_result,ignore_audio_couple)
            delay_second_method = self.second_delay_test(mean_between_delay,ignore_audio_couple)
            self.video_obj_1.extract_audio_in_part(self.language,self.audioParam,cutTime=self.list_cut_begin_length,asDefault=True)
        except Exception as e:
            self.video_obj_1.extract_audio_in_part(self.language,self.audioParam,cutTime=self.list_cut_begin_length,asDefault=True)
            sys.stderr.write("We get an error during adjuster_chroma_bugged:\n"+str(e)+"\n")
            with errors_merge_lock:
                errors_merge.append("We get an error during adjuster_chroma_bugged:\n"+str(e)+"\n")
            return None
    
        calculated_delay = mean_between_delay+round(delay_second_method*1000) #delay_first_method+round(delay_second_method*1000)
        if abs(delay_second_method) < 0.125:
            # calculated_delay-delay_first_method_lower_result < 125 and calculated_delay-delay_first_method_lower_result > 0:
            sys.stderr.write(f"The delay {calculated_delay} find with adjuster_chroma_bugged is valid for {self.video_obj_1.filePath} and {self.video_obj_2.filePath}. The original delay was between {delay_first_method_lower_result} and {delay_first_method_bigger_result} \n")
            with errors_merge_lock:
                errors_merge.append(f"The delay {calculated_delay} find with adjuster_chroma_bugged is valid for {self.video_obj_1.filePath} and {self.video_obj_2.filePath}. The original delay was between {delay_first_method_lower_result} and {delay_first_method_bigger_result} \n")
            return calculated_delay
        else:
            sys.stderr.write(f"The delay {calculated_delay} find with adjuster_chroma_bugged is not valid for {self.video_obj_1.filePath} and {self.video_obj_2.filePath}. The original delay was between {delay_first_method_lower_result} and {delay_first_method_bigger_result} \n")
            with errors_merge_lock:
                errors_merge.append(f"The delay {calculated_delay} find with adjuster_chroma_bugged is not valid for {self.video_obj_1.filePath} and {self.video_obj_2.filePath}. The original delay was between {delay_first_method_lower_result} and {delay_first_method_bigger_result} \n")
            return None
        
    def get_delays_dict(self,delay_Fidelity_Values,delayUse=0):
        delays_dict = {}
        for key_audio, delay_fidelity_list in delay_Fidelity_Values.items():
            delays_dict[key_audio] = [delayUse + delay_fidelity[2] for delay_fidelity in delay_fidelity_list]
        return delays_dict
    
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
            if len(set_delay) == 1 or variance(set_delay) < max_delay_variance_second_method:
                delay_detected.update(set_delay)
            elif abs(delay_list[0][1]-delay_list[-1][1]) < max_delay_variance_second_method:
                sys.stderr.write(f"Variance delay in the second test is to big {set_delay} with {self.video_obj_1.filePath} and {self.video_obj_2.filePath} ")
                delay_detected.add(delay_list[0][1])
                delay_detected.add(delay_list[-1][1])
            else:
                raise Exception(f"Variance delay in the second test is to big {set_delay} with {self.video_obj_1.filePath} and {self.video_obj_2.filePath} but the first and last part have the similar delay\n")
        
        if len(delay_detected) != 1 and variance(delay_detected) > max_delay_variance_second_method:
            raise Exception(f"Multiple delay found with the method 2 and in test 1 {delay_detected} for {self.video_obj_1.filePath} and {self.video_obj_2.filePath} at the second method")
        else:
            '''
                TODO:
                    protect the memory to overload
            '''
            self.video_obj_1.extract_audio_in_part(self.language,self.audioParam,cutTime=[[strftime('%H:%M:%S',gmtime(int(self.begin_in_second))),strftime('%H:%M:%S',gmtime(int(self.lenghtTime*video.number_cut/cut_file_to_get_delay_second_method)))]])
            begining_in_second, begining_in_millisecond = video.get_begin_time_with_millisecond(delayUse,self.begin_in_second)
            self.video_obj_2.extract_audio_in_part(self.language,self.audioParam,cutTime=[[strftime('%H:%M:%S',gmtime(begining_in_second))+begining_in_millisecond,strftime('%H:%M:%S',gmtime(int(self.lenghtTime*video.number_cut/cut_file_to_get_delay_second_method)))]])
            self.video_obj_1.wait_end_ffmpeg_progress_audio()
            self.video_obj_2.wait_end_ffmpeg_progress_audio()
            for i in range(0,len(self.video_obj_1.tmpFiles['audio'])):
                for j in range(0,len(self.video_obj_2.tmpFiles['audio'])):
                    if f"{i}-{j}" not in ignore_audio_couple:
                        delay_between_two_audio = []
                        delay_Values[f"{i}-{j}"] = delay_between_two_audio
                        delay_between_two_audio.append(second_correlation(self.video_obj_1.tmpFiles['audio'][i][0],self.video_obj_2.tmpFiles['audio'][j][0]))
            
            import gc
            gc.collect()
            delay_detected = []
            for key_audio, delay_list in delay_Values.items():
                for delay in delay_list:
                    delay_detected.append(delay[1])
            return mean(delay_detected)
            
    def get_best_video(self,delay):
        delay,begins_video_for_compare_quality = video.get_good_frame(self.video_obj_1, self.video_obj_2, self.begin_in_second, self.lenghtTime, self.time_by_test_best_quality_converted, (delay/1000))

        if video.get_best_quality_video(self.video_obj_1, self.video_obj_2, begins_video_for_compare_quality, self.time_by_test_best_quality_converted) == 1:
            self.video_obj_1.extract_audio_in_part(self.language,self.audioParam,cutTime=self.list_cut_begin_length,asDefault=True)
            self.video_obj_2.remove_tmp_files(type_file="audio")
            self.video_obj_with_best_quality = self.video_obj_1
            delay = self.adjust_delay_to_frame(delay)
            self.video_obj_2.delays[self.language] += (delay*-1.0) # Delay you need to give to mkvmerge to be good.
        else:
            self.video_obj_2.extract_audio_in_part(self.language,self.audioParam,cutTime=self.list_cut_begin_length,asDefault=True)
            self.video_obj_1.remove_tmp_files(type_file="audio")
            self.video_obj_with_best_quality = self.video_obj_2
            delay = self.adjust_delay_to_frame(delay)
            self.video_obj_1.delays[self.language] += delay # Delay you need to give to mkvmerge to be good.
            
    def adjust_delay_to_frame(self,delay):
        if self.video_obj_with_best_quality.video["FrameRate_Mode"] == "CFR":
            getcontext().prec = 10
            framerate = Decimal(self.video_obj_with_best_quality.video["FrameRate"])
            number_frame = round(Decimal(delay)/framerate)
            distance_frame = Decimal(delay)%framerate
            if abs(distance_frame) < framerate/Decimal(2.0):
                return Decimal(number_frame)*framerate
            elif number_frame > 0:
                return Decimal(number_frame+1)*framerate
            elif number_frame < 0:
                return Decimal(number_frame-1)*framerate
            elif distance_frame > 0:
                return Decimal(number_frame+1)*framerate
            elif distance_frame < 0:
                return Decimal(number_frame-1)*framerate
            else:
                return delay
            
        else:
            ''' TODO:
                ADD VFR calculation if found'''
            return delay

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

class get_cut_time(Thread):
    '''
    classdocs
    '''


    def __init__(self, main_video_obj,video_obj_to_cut,begin_in_second,audioParam,language,lenghtTime,lenghtTimePrepare,list_cut_begin_length,time_by_test_best_quality_converted):
        '''
        Constructor
        '''
        Thread.__init__(self)
        self.main_video_obj = main_video_obj
        self.video_obj_to_cut = video_obj_to_cut
        self.begin_in_second = begin_in_second
        self.audioParam = audioParam
        self.language = language
        self.lenghtTime = lenghtTime
        self.lenghtTimePrepare = lenghtTimePrepare
        self.list_cut_begin_length = list_cut_begin_length
        self.time_by_test_best_quality_converted = time_by_test_best_quality_converted

    def run(self):
        try:
            delay = self.get_first_delay_and_gap()
            if self.process_to_get_best_video:
                self.get_best_video(delay)
            else: # You must have the video you want process in video_obj_1
                self.video_obj_1.extract_audio_in_part(self.language,self.audioParam,cutTime=self.list_cut_begin_length,asDefault=True)
                self.video_obj_2.remove_tmp_files(type_file="audio")
                self.video_obj_with_best_quality = self.video_obj_1
                self.video_obj_2.delays[self.language] += (delay*-1.0) # Delay you need to give to mkvmerge to be good.
        except Exception as e:
            traceback.print_exc()
            sys.stderr.write(str(e)+"\n")
    
    def get_first_delay_and_gap(self):
        delay_Fidelity_Values = get_delay_fidelity(self.main_video_obj,self.video_obj_to_cut,self.lenghtTime)
        # Il va falloir verifier que nous avons bien les mêmes delays entre les différents audios
        keys_audio = list(delay_Fidelity_Values.keys())
        values_of_delay = delay_Fidelity_Values[keys_audio[0]]
        for key_audio, delay_fidelity_list in delay_Fidelity_Values.items():
            for i in range(len(values_of_delay)):
                if values_of_delay[i] != delay_fidelity_list[i]:
                    raise Exception(f"{delay_Fidelity_Values} Impossible to find a way to cut {self.video_obj_to_cut.filePath} who have differents audio not compatible with {self.main_video_obj.filePath}")

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

def prepare_get_delay_sub(videos_obj,language):
    audio_parameter_to_use_for_comparison = {'Format':"WAV",
                                             'codec':"pcm_s16le",
                                             'Channels':"2"}
    min_channel = video.get_less_channel_number(videos_obj,language)
    if min_channel == "1":
        audio_parameter_to_use_for_comparison['Channels'] = min_channel

    min_video_duration_in_sec = video.get_shortest_audio_durations(videos_obj,language)
    get_good_parameters_to_get_fidelity(videos_obj,language,audio_parameter_to_use_for_comparison,min_video_duration_in_sec)
    
    begin_in_second,length_time = video.generate_begin_and_length_by_segment(min_video_duration_in_sec)
    length_time_converted = strftime('%H:%M:%S',gmtime(length_time))
    list_cut_begin_length = video.generate_cut_with_begin_length(begin_in_second,length_time,length_time_converted)

    return begin_in_second,audio_parameter_to_use_for_comparison,length_time,length_time_converted,list_cut_begin_length

def prepare_get_delay(videos_obj,language,audioRules):
    begin_in_second,audio_parameter_to_use_for_comparison,length_time,length_time_converted,list_cut_begin_length = prepare_get_delay_sub(videos_obj,language)
    for videoObj in videos_obj:
        for language_obj,audios in videoObj.audios.items():
            for audio in audios:
                audio["keep"] = True
        videoObj.extract_audio_in_part(language,audio_parameter_to_use_for_comparison,cutTime=list_cut_begin_length)
        videoObj.delays[language] = 0
        for language_obj,audios in videoObj.commentary.items():
            for audio in audios:
                audio["keep"] = (not tools.special_params["remove_commentary"])
    
    return begin_in_second,audio_parameter_to_use_for_comparison,length_time,length_time_converted,list_cut_begin_length

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

def find_a_cut_for_not_compatible(list_not_compatible_video,dict_file_path_obj,main_video,videosObj,language,audioRules):
    if video.number_cut < 15:
        video.number_cut = 15
    elif (video.number_cut % 2) == 0:
        video.number_cut += 1
    
    begin_in_second,worseAudioQualityWillUse,length_time,length_time_converted,list_cut_begin_length = prepare_get_delay(videosObj,language,audioRules)
    dict_file_path_obj[main_video].extract_audio_in_part(language,worseAudioQualityWillUse,cutTime=list_cut_begin_length)
    for not_compatible_video in list_not_compatible_video:
        if not_compatible_video in dict_file_path_obj:
            dict_file_path_obj[not_compatible_video].extract_audio_in_part(language,worseAudioQualityWillUse,cutTime=list_cut_begin_length)
    
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
    if len(videosObj):
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
    else:
        already_compared = {forced_best_video:{}}
    
    return already_compared

def find_differences_and_keep_best_audio(video_obj,language,audioRules):
    try:
        begin_in_second,audio_parameter_to_use_for_comparison,length_time,length_time_converted,list_cut_begin_length = prepare_get_delay_sub([video_obj],language)
        videoObj.extract_audio_in_part(language,audio_parameter_to_use_for_comparison,cutTime=list_cut_begin_length)

        from sys import stderr
        ignore_compare = set([f"{i}-{i}" for i in range(len(video_obj.audios[language]))])
        for i in range(len(video_obj.audios[language])):
            for j in range(i+1,len(video_obj.audios[language])):
                ignore_compare.add(f"{j}-{i}")
        delay_Fidelity_Values = get_delay_fidelity(video_obj,video_obj,lenghtTime,ignore_audio_couple=ignore_compare)
        
        fileid_audio = {}
        validation = {}
        for audio in video_obj.audios[language]:
            fileid_audio[audio["audio_pos_file"]] = audio
            validation[audio["audio_pos_file"]] = {}

        to_compare = []
        for i in range(len(video_obj.audios[language])):
            to_compare.append(i)
            for j in range(i+1,len(video_obj.audios[language])):
                set_delay = set()
                for delay_fidelity in delay_Fidelity_Values[f"{i}-{j}"]:
                    set_delay.add(delay_fidelity[2])
                if len(set_delay) == 1 and abs(list(set_delay)[0]) < 128:
                    validation[i][j] = True
                elif len(set_delay) == 1 and abs(list(set_delay)[0]) >= 128:
                    validation[i][j] = False
                    stderr.write(f"Be carreful find_differences_and_keep_best_audio on {language} find a delay of {set_delay}\n")
                else:
                    validation[i][j] = False
        
        while len(to_compare):
            main = to_compare.pop(0)
            list_compatible = set()
            not_compatible = set()
            for i in validation[main].keys():
                if validation[main][i] and i not in not_compatible:
                    list_compatible.add(i)
                    for j in validation[i].keys():
                        if (not(validation[i][j])):
                            not_compatible.add(j)
            list_compatible = list_compatible - not_compatible
            if len(list_compatible):
                list_audio_metadata_compatible = [fileid_audio[main]]
                for id_audio in list_compatible:
                    list_audio_metadata_compatible.append(fileid_audio[id_audio])
                keep_best_audio(list_audio_metadata_compatible,audioRules)
                to_compare = to_compare - list_compatible
            
    except Exception as e:
        stderr.write(f"Error processing find_differences_and_keep_best_audio on {language}: {e}\n")
    finally:
        video_obj.remove_tmp_files(type_file="audio")

def keep_best_audio(list_audio_metadata,audioRules):
    '''
    Todo:
        Integrate https://github.com/Sg4Dylan/FLAD/tree/main
    '''
    for i,audio_1 in enumerate(list_audio_metadata):
        for j,audio_2 in enumerate(list_audio_metadata):
            if audio_1['Format'].lower() == audio_2['Format'].lower():
                try:
                    if float(audio_1['Channels']) == float(audio_2['Channels']):
                        if int(audio_1['SamplingRate']) >= int(audio_2['SamplingRate']) and int(video.get_bitrate(audio_1)) > int(video.get_bitrate(audio_2)):
                            audio_2['keep'] = False
                        elif int(audio_2['SamplingRate']) >= int(audio_1['SamplingRate']) and int(video.get_bitrate(audio_2)) > int(video.get_bitrate(audio_1)):
                            audio_1['keep'] = False
                    elif float(audio_1['Channels']) > float(audio_2['Channels']):
                        if int(audio_1['SamplingRate']) >= int(audio_2['SamplingRate']) and (float(video.get_bitrate(audio_1))/float(audio_1['Channels'])) > (float(video.get_bitrate(audio_2))/float(audio_2['Channels'])*0.95):
                            audio_2['keep'] = False
                    elif float(audio_2['Channels']) > float(audio_1['Channels']):
                        if int(audio_2['SamplingRate']) >= int(audio_1['SamplingRate']) and (float(video.get_bitrate(audio_2))/float(audio_2['Channels'])) > (float(video.get_bitrate(audio_1))/float(audio_1['Channels'])*0.95):
                            audio_1['keep'] = False
                except Exception as e:
                    sys.stderr.write(str(e))
            else:
                if audio_1['Format'].lower() in audioRules:
                    if audio_2['Format'].lower() in audioRules[audio_1['Format'].lower()]:
                        try:
                            if int(audio_1['SamplingRate']) >= int(audio_2['SamplingRate']) and float(audio_1['Channels']) >= float(audio_2['Channels']):
                                if isinstance(audioRules[audio_1['Format'].lower()][audio_2['Format'].lower()], bool):
                                    if audioRules[audio_1['Format'].lower()][audio_2['Format'].lower()]:
                                        audio_2['keep'] = False
                                elif isinstance(audioRules[audio_1['Format'].lower()][audio_2['Format'].lower()], float):
                                    if int(video.get_bitrate(audio_1)) > int(video.get_bitrate(audio_2))*audioRules[audio_1['Format'].lower()][audio_2['Format'].lower()]:
                                        audio_2['keep'] = False
                        except Exception as e:
                            sys.stderr.write(str(e))
                        
                        try:
                            if int(audio_2['SamplingRate']) >= int(audio_1['SamplingRate']) and float(audio_2['Channels']) >= float(audio_1['Channels']):
                                if isinstance(audioRules[audio_2['Format'].lower()][audio_1['Format'].lower()], bool):
                                    if audioRules[audio_2['Format'].lower()][audio_1['Format'].lower()]:
                                        audio_1['keep'] = False
                                elif isinstance(audioRules[audio_2['Format'].lower()][audio_1['Format'].lower()], float):
                                    if int(video.get_bitrate(audio_2)) > int(video.get_bitrate(audio_1))*audioRules[audio_2['Format'].lower()][audio_1['Format'].lower()]:
                                        audio_1['keep'] = False
                        except Exception as e:
                            sys.stderr.write(str(e))

def remove_sub_language(video_sub_track_list,language,number_sub_will_be_copy,number_max_sub_stream):
    if number_sub_will_be_copy > number_max_sub_stream:
        for sub in video_sub_track_list[language]:
            if (sub['keep']) and number_sub_will_be_copy > number_max_sub_stream:
                sub['keep'] = False
                number_sub_will_be_copy -= 1
    return number_sub_will_be_copy

def keep_one_ass(groupID_srt_type_in,number_sub_will_be_copy,number_max_sub_stream):
    for ass_name in ["forced_ass","hi_ass","dub_ass","ass"]:
        if number_sub_will_be_copy > number_max_sub_stream:
            for comparative_sub in groupID_srt_type_in.values():
                if ass_name in comparative_sub and len(comparative_sub[ass_name]) > 1:
                    for i in range(1,len(comparative_sub[ass_name])):
                        comparative_sub[ass_name][i]['keep'] = False
                    number_sub_will_be_copy -= (len(comparative_sub[ass_name]) - 1)
    return number_sub_will_be_copy

def sub_group_id_detector_and_clean_srt_when_ass_with_test(video_sub_track_list,language,language_groupID_srt_type_in,number_sub_will_be_copy,number_max_sub_stream):
    if number_sub_will_be_copy > number_max_sub_stream:
        sub_group_id_detector(video_sub_track_list[language],tools.group_title_sub[language],language_groupID_srt_type_in[language])

        if number_sub_will_be_copy > number_max_sub_stream:
            number_sub_will_be_copy = clean_srt_when_ass(language_groupID_srt_type_in[language],"hi_ass","hi_srt",number_sub_will_be_copy)
        if number_sub_will_be_copy > number_max_sub_stream:
            number_sub_will_be_copy = clean_srt_when_ass(language_groupID_srt_type_in[language],"dub_ass","dub_srt",number_sub_will_be_copy)
        if number_sub_will_be_copy > number_max_sub_stream:
            number_sub_will_be_copy = clean_srt_when_ass(language_groupID_srt_type_in[language],"ass","srt",number_sub_will_be_copy)
    return number_sub_will_be_copy

def sub_group_id_detector(sub_list,group_title_sub_for_language,groupID_srt_type_in):
    for sub in sub_list:
        if (sub['keep']):
            if codec in tools.sub_type_near_srt:
                if test_if_hearing_impaired(sub):
                    insert_type_in_group_sub_title(clean_hearing_impaired_title(sub),"hi_srt",group_title_sub_for_language,groupID_srt_type_in,sub)
                elif test_if_dubtitle(sub):
                    insert_type_in_group_sub_title(clean_dubtitle_title(sub),"dub_srt",group_title_sub_for_language,groupID_srt_type_in,sub)
                elif (not test_if_forced(sub)):
                    insert_type_in_group_sub_title(clean_title(sub),"srt",group_title_sub_for_language,groupID_srt_type_in,sub)

            elif codec not in tools.sub_type_not_encodable:
                if test_if_hearing_impaired(sub):
                    insert_type_in_group_sub_title(clean_hearing_impaired_title(sub),"hi_ass",group_title_sub_for_language,groupID_srt_type_in,sub)
                elif test_if_dubtitle(sub):
                    insert_type_in_group_sub_title(clean_dubtitle_title(sub),"dub_ass",group_title_sub_for_language,groupID_srt_type_in,sub)
                elif (not test_if_forced(sub)):
                    insert_type_in_group_sub_title(clean_title(sub),"ass",group_title_sub_for_language,groupID_srt_type_in,sub)

def clean_srt_when_ass(groupID_srt_type_in,ass_name,srt_name,number_sub_will_be_copy):
    for comparative_sub in groupID_srt_type_in.values():
        if ass_name in comparative_sub and len(comparative_sub[ass_name]) and srt_name in comparative_sub and len(comparative_sub[srt_name]):
            for sub in comparative_sub[srt_name]:
                sub['keep'] = False
            number_sub_will_be_copy -= len(comparative_sub[srt_name])
        elif srt_name in comparative_sub and len(comparative_sub[srt_name]) > 1:
            for i in range(1,len(comparative_sub[srt_name])):
                comparative_sub[srt_name][i]['keep'] = False
            number_sub_will_be_copy -= (len(comparative_sub[srt_name]) - 1)
    return number_sub_will_be_copy

def get_sub_title_group_id(groups,sub_title):
    for i,group in enumerate(groups):
        if sub_title in group:
            return i
    return None

def insert_type_in_group_sub_title(sub_clean_title,type_sub,groups,groupID_srt_type_in,sub):
    group_id = get_sub_title_group_id(groups,sub_clean_title)
    if group_id == None:
        groups.append([sub_clean_title])
        group_id = len(groups)-1
    
    if group_id not in groupID_srt_type_in:
        groupID_srt_type_in[group_id] = {}
    if type_sub not in groupID_srt_type_in[group_id]:
        groupID_srt_type_in[group_id][type_sub] = [sub]
    else:
        groupID_srt_type_in[group_id][type_sub].append(sub)

def clean_title(sub):
    clean_title = ""
    if "Title" in sub:
        clean_title = re.sub(r'^\s*',"",clean_title)
        clean_title = re.sub(r'\s*$',"",clean_title)
    return clean_title

def clean_dubtitle_title(sub):
    clean_title = ""
    if "Title" in sub:
        clean_title = re.sub(r'\s*\({0,1}dubtitle\){0,1}\s*',"",sub["Title"].lower())
        clean_title = re.sub(r'^\s*',"",clean_title)
        clean_title = re.sub(r'\s*$',"",clean_title)
    return clean_title

def test_if_dubtitle(sub):
    if "Title" in sub and re.match(r".*dubtitle.*", sub["Title"].lower()):
        return True
    return False

def clean_hearing_impaired_title(sub):
    clean_title = ""
    if "Title" in sub:
        if re.match(r".*sdh.*", sub["Title"].lower()):
            clean_title = re.sub(r'\s*\({0,1}sdh\){0,1}\s*',"",sub["Title"].lower())
        elif re.match(r".*\(cc\).*", sub["Title"].lower()):
            clean_title = re.sub(r'\s*\(cc\)\s*',"",sub["Title"].lower())
        elif 'hi' == sub["Title"].lower() or 'cc' == sub["Title"].lower():
            clean_title = ""
        else:
            clean_title = sub["Title"].lower()
        clean_title = re.sub(r'^\s*',"",clean_title)
        clean_title = re.sub(r'\s*$',"",clean_title)
    return clean_title

def test_if_hearing_impaired(sub):
    if "Title" in sub:
        if re.match(r".*sdh.*", sub["Title"].lower()) or 'cc' == sub["Title"].lower() or 'hi' == sub["Title"].lower() or re.match(r".*\(cc\).*", sub["Title"].lower()):
            return True
    if ("flag_hearing_impaired" in sub['properties'] and sub['properties']["flag_hearing_impaired"]):
        return True
    return False

def clean_forced_title(sub):
    clean_title = ""
    if "Title" in sub:
        clean_title = re.sub(r'\s*\({0,1}forced\){0,1}\s*',"",sub["Title"].lower())
        clean_title = re.sub(r'^\s*',"",clean_title)
        clean_title = re.sub(r'\s*$',"",clean_title)
    return clean_title

def test_if_forced(sub):
    if "Title" in sub and re.match(r".*forced.*", sub["Title"].lower()):
        return True
    return False

def clean_number_stream_to_be_lover_than_max(number_max_sub_stream,video_sub_track_list):
    try:
        unique_md5 = set()
        number_sub_will_be_copy = 0
        for language,subs in video_sub_track_list.items():
            for sub in subs:
                if sub['keep']:
                    if sub['MD5'] not in unique_md5:
                        number_sub_will_be_copy += 1
                        if sub['MD5'] != '':
                            unique_md5.add(sub['MD5'])
                    else:
                        sub['keep'] = False
        
        if number_sub_will_be_copy > number_max_sub_stream:
            language_groupID_srt_type_in = {}
            # Remove forced srt sub if we have an ass.
            for language,subs in video_sub_track_list.items():
                if language not in tools.group_title_sub:
                    tools.group_title_sub[language] = []
                groupID_srt_type_in = {}
                language_groupID_srt_type_in[language] = groupID_srt_type_in
                for sub in subs:
                    if (sub['keep']):
                        if codec in tools.sub_type_near_srt and test_if_forced(sub):
                            insert_type_in_group_sub_title(clean_forced_title(sub),"forced_srt",tools.group_title_sub[language],groupID_srt_type_in,sub)
                        elif codec not in tools.sub_type_not_encodable and test_if_forced(sub):
                            insert_type_in_group_sub_title(clean_forced_title(sub),"forced_ass",tools.group_title_sub[language],groupID_srt_type_in,sub)
                number_sub_will_be_copy = clean_srt_when_ass(groupID_srt_type_in,"forced_ass","forced_srt",number_sub_will_be_copy)

            # Remove srt sub on not keep
            if number_sub_will_be_copy > number_max_sub_stream:
                language_to_clean = video_sub_track_list.keys() - tools.language_to_keep - tools.language_to_try_to_keep
                for language in language_to_clean:
                    sub_group_id_detector(video_sub_track_list[language],tools.group_title_sub[language],language_groupID_srt_type_in[language])

                    number_sub_will_be_copy = clean_srt_when_ass(language_groupID_srt_type_in[language],"hi_ass","hi_srt",number_sub_will_be_copy)
                    number_sub_will_be_copy = clean_srt_when_ass(language_groupID_srt_type_in[language],"dub_ass","dub_srt",number_sub_will_be_copy)
                    number_sub_will_be_copy = clean_srt_when_ass(language_groupID_srt_type_in[language],"ass","srt",number_sub_will_be_copy)
                
                if number_sub_will_be_copy > number_max_sub_stream:
                    for language in tools.language_to_try_to_keep:
                        number_sub_will_be_copy = sub_group_id_detector_and_clean_srt_when_ass_with_test(video_sub_track_list,language,language_groupID_srt_type_in,number_sub_will_be_copy,number_max_sub_stream)
                    
                    if number_sub_will_be_copy > number_max_sub_stream:
                        for language in language_to_clean:
                            if number_sub_will_be_copy > number_max_sub_stream:
                                number_sub_will_be_copy = keep_one_ass(language_groupID_srt_type_in[language],number_sub_will_be_copy,number_max_sub_stream)
                        
                        if number_sub_will_be_copy > number_max_sub_stream:
                            for language in tools.language_to_keep:
                                number_sub_will_be_copy = sub_group_id_detector_and_clean_srt_when_ass_with_test(video_sub_track_list,language,language_groupID_srt_type_in,number_sub_will_be_copy,number_max_sub_stream)
                            
                            if number_sub_will_be_copy > number_max_sub_stream:
                                for language in language_to_clean:
                                    number_sub_will_be_copy = remove_sub_language(video_sub_track_list,language,number_sub_will_be_copy,number_max_sub_stream)
                                
                                if number_sub_will_be_copy > number_max_sub_stream:
                                    for language in tools.language_to_try_to_keep:
                                        number_sub_will_be_copy = keep_one_ass(language_groupID_srt_type_in[language],number_sub_will_be_copy,number_max_sub_stream)

    except Exception as e:
        stderr.write(f"Error processing clean_number_stream_to_be_lover_than_max: {e}\n")

def not_keep_ass_converted_in_srt(file_path,keep_sub_ass,keep_sub_srt):
    set_md5_ass = set()
    for sub in keep_sub_ass:
        stream_ID,md5 = video.subtitle_text_srt_md5(file_path,sub["StreamOrder"])
        set_md5_ass.add(md5)
    for sub in keep_sub_srt:
        stream_ID,md5 = video.subtitle_text_srt_md5(file_path,sub["StreamOrder"])
        if md5 in set_md5_ass:
            sub['keep'] = False

def generate_merge_command_insert_ID_sub_track_set_not_default(merge_cmd,video_sub_track_list,md5_sub_already_added,list_track_order=[]):
    track_to_remove = set()
    number_track_sub = 0
    dic_language_list_track_ID = {}
    for language,subs in video_sub_track_list.items():
        for sub in subs:
            if (sub['keep'] and sub['MD5'] not in md5_sub_already_added):
                number_track_sub += 1
                if sub['MD5'] != '':
                    md5_sub_already_added.add(sub['MD5'])
                
                codec = sub['ffprobe']["codec_name"].lower()
                if codec in tools.sub_type_not_encodable:
                    language_and_type = language+'uncodable'
                elif codec in tools.sub_type_near_srt:
                    language_and_type = language+'srt'
                else:
                    language_and_type = language+'all'
                merge_cmd.extend(["--default-track-flag", sub["StreamOrder"]+":0"])
                if "Title" in sub:
                    if re.match(r".*forced.*", sub["Title"].lower()):
                        merge_cmd.extend(["--forced-display-flag", sub["StreamOrder"]+":1"])
                        if language_and_type+'_forced' not in dic_language_list_track_ID:
                            dic_language_list_track_ID[language_and_type+'_forced'] = [sub["StreamOrder"]]
                        else:
                            dic_language_list_track_ID[language_and_type+'_forced'].append(sub["StreamOrder"])
                    elif re.match(r".*sdh.*", sub["Title"].lower()) or 'cc' == sub["Title"].lower() or 'hi' == sub["Title"].lower() or re.match(r".*\(cc\).*", sub["Title"].lower()) or ("flag_hearing_impaired" in sub['properties'] and sub['properties']["flag_hearing_impaired"]):
                        merge_cmd.extend(["--hearing-impaired-flag", sub["StreamOrder"]+":1"])
                        if language_and_type+'_hearing' not in dic_language_list_track_ID:
                            dic_language_list_track_ID[language_and_type+'_hearing'] = [sub["StreamOrder"]]
                        else:
                            dic_language_list_track_ID[language_and_type+'_hearing'].append(sub["StreamOrder"])
                    elif re.match(r".*dubtitle.*", sub["Title"].lower()):
                        if language_and_type+'_dubtitle' not in dic_language_list_track_ID:
                            dic_language_list_track_ID[language_and_type+'_dubtitle'] = [sub["StreamOrder"]]
                        else:
                            dic_language_list_track_ID[language_and_type+'_dubtitle'].append(sub["StreamOrder"])
                    else:
                        if language_and_type not in dic_language_list_track_ID:
                            dic_language_list_track_ID[language_and_type] = [sub["StreamOrder"]]
                        else:
                            dic_language_list_track_ID[language_and_type].append(sub["StreamOrder"])
                else:
                    if language_and_type not in dic_language_list_track_ID:
                        dic_language_list_track_ID[language_and_type] = [sub["StreamOrder"]]
                    else:
                        dic_language_list_track_ID[language_and_type].append(sub["StreamOrder"])
            else:
                track_to_remove.add(sub["StreamOrder"])
    if len(track_to_remove):
        merge_cmd.extend(["-s","!"+",".join(track_to_remove)])
    
    for language in sorted(dic_language_list_track_ID.keys()):
        list_track_order.extend(dic_language_list_track_ID[language])
    
    return number_track_sub

def generate_merge_command_insert_ID_audio_track_to_remove_and_new_und_language_set_not_default_not_forced(merge_cmd,audio):
    merge_cmd.extend(["--forced-display-flag", audio["StreamOrder"]+":0", "--default-track-flag", audio["StreamOrder"]+":0"])

default_audio = True
def generate_merge_command_insert_ID_audio_track_to_remove_and_new_und_language(merge_cmd,video_audio_track_list,video_commentary_track_list,video_audio_desc_track_list,md5_audio_already_added,list_track_order=[]):
    global default_audio
    number_track_audio = 0
    dic_language_list_track_ID = {}
    if len(video_audio_track_list) == 2 and "und" in video_audio_track_list and tools.default_language_for_undetermine != "und":
        # This step is linked by the fact if you have und audio they are orginialy convert in another language
        # This was convert in a language, but the object is the same and can be compared
        if video_audio_track_list[tools.default_language_for_undetermine] == video_audio_track_list['und']:
            del video_audio_track_list[tools.default_language_for_undetermine]
        
    track_to_remove = set()
    for language,audios in video_audio_track_list.items():
        for audio in audios:
            if ((not audio["keep"]) or (audio["MD5"] != '' and audio["MD5"] in md5_audio_already_added)):
                track_to_remove.add(audio["StreamOrder"])
            else:
                number_track_audio += 1
                if language not in dic_language_list_track_ID:
                    dic_language_list_track_ID[language] = [audio["StreamOrder"]]
                else:
                    dic_language_list_track_ID[language].append(audio["StreamOrder"])
                md5_audio_already_added.add(audio["MD5"])
                original_audio = False
                if language == "und" and tools.special_params["change_all_und"]:
                    merge_cmd.extend(["--language", audio["StreamOrder"]+":"+tools.default_language_for_undetermine])
                    if tools.default_language_for_undetermine == tools.special_params["original_language"]:
                        merge_cmd.extend(["--original-flag", audio["StreamOrder"]])
                        original_audio = True
                elif language == tools.special_params["original_language"]:
                    merge_cmd.extend(["--original-flag", audio["StreamOrder"]])
                    original_audio = True
                if default_audio and original_audio:
                    merge_cmd.extend(["--forced-display-flag", audio["StreamOrder"]+":0", "--default-track-flag", audio["StreamOrder"]+":1"])
                    default_audio = False
                else:
                    generate_merge_command_insert_ID_audio_track_to_remove_and_new_und_language_set_not_default_not_forced(merge_cmd,audio)
    for language,audios in video_commentary_track_list.items():
        for audio in audios:
            if ((not audio["keep"]) or (audio["MD5"] != '' and audio["MD5"] in md5_audio_already_added)):
                track_to_remove.add(audio["StreamOrder"])
            else:
                number_track_audio += 1
                if language+'_com' not in dic_language_list_track_ID:
                    dic_language_list_track_ID[language+'_com'] = [audio["StreamOrder"]]
                else:
                    dic_language_list_track_ID[language+'_com'].append(audio["StreamOrder"])
                md5_audio_already_added.add(audio["MD5"])
                if language == "und" and tools.special_params["change_all_und"]:
                    merge_cmd.extend(["--language", audio["StreamOrder"]+":"+tools.default_language_for_undetermine])
                generate_merge_command_insert_ID_audio_track_to_remove_and_new_und_language_set_not_default_not_forced(merge_cmd,audio)
                merge_cmd.extend(["--commentary-flag", audio["StreamOrder"]])
    for language,audios in video_audio_desc_track_list.items():
        for audio in audios:
            if (audio["MD5"] in md5_audio_already_added):
                track_to_remove.add(audio["StreamOrder"])
            else:
                number_track_audio += 1
                if language+'_visuali' not in dic_language_list_track_ID:
                    dic_language_list_track_ID[language+'_visuali'] = [audio["StreamOrder"]]
                else:
                    dic_language_list_track_ID[language+'_visuali'].append(audio["StreamOrder"])
                md5_audio_already_added.add(audio["MD5"])
                if language == "und" and tools.special_params["change_all_und"]:
                    merge_cmd.extend(["--language", audio["StreamOrder"]+":"+tools.default_language_for_undetermine])
                generate_merge_command_insert_ID_audio_track_to_remove_and_new_und_language_set_not_default_not_forced(merge_cmd,audio)
                merge_cmd.extend(["--visual-impaired-flag", audio["StreamOrder"]])

    if len(track_to_remove):
        merge_cmd.extend(["-a","!"+",".join(track_to_remove)])
        
    for language in sorted(dic_language_list_track_ID.keys()):
        list_track_order.extend(dic_language_list_track_ID[language])
    
    return number_track_audio

def generate_merge_command_common_md5(video_obj,delay_to_put,ffmpeg_cmd_dict,md5_audio_already_added,md5_sub_already_added,duration_best_video):
    number_track = generate_new_file(video_obj,delay_to_put,ffmpeg_cmd_dict,md5_audio_already_added,md5_sub_already_added,duration_best_video)
    if number_track:
        ffmpeg_cmd_dict['metadata_cmd'].extend(["-A", "-S", "-D", "--no-chapters", video_obj.filePath])
    else:
        if delay_to_put != 0:
            ffmpeg_cmd_dict['metadata_cmd'].extend(["--sync", f"-1:{round(delay_to_put)}"])
        ffmpeg_cmd_dict['metadata_cmd'].extend(["-A", "-S", "-D", video_obj.filePath])
    
    print(f'\t{video_obj.filePath} will add with a delay of {delay_to_put}')
    
    for video_obj_common_md5 in video_obj.sameAudioMD5UseForCalculation:
        generate_merge_command_common_md5(video_obj_common_md5,delay_to_put,ffmpeg_cmd_dict,md5_audio_already_added,md5_sub_already_added,duration_best_video)

def generate_merge_command_other_part(video_path_file,dict_list_video_win,dict_file_path_obj,ffmpeg_cmd_dict,delay_winner,common_language_use_for_generate_delay,md5_audio_already_added,md5_sub_already_added,duration_best_video):
    video_obj = dict_file_path_obj[video_path_file]
    delay_to_put = video_obj.delays[common_language_use_for_generate_delay] + delay_winner
    number_track = generate_new_file(video_obj,delay_to_put,ffmpeg_cmd_dict,md5_audio_already_added,md5_sub_already_added,duration_best_video)
    if number_track:
        ffmpeg_cmd_dict['metadata_cmd'].extend(["-A", "-S", "-D", "--no-chapters", video_obj.filePath])
    else:
        if delay_to_put != 0:
            ffmpeg_cmd_dict['metadata_cmd'].extend(["--sync", f"-1:{round(delay_to_put)}"])
        ffmpeg_cmd_dict['metadata_cmd'].extend(["-A", "-S", "-D", video_obj.filePath])
    
    print(f'\t{video_obj.filePath} will add with a delay of {delay_to_put}')
    
    for video_obj_common_md5 in video_obj.sameAudioMD5UseForCalculation:
        generate_merge_command_common_md5(video_obj_common_md5,delay_to_put,ffmpeg_cmd_dict,md5_audio_already_added,md5_sub_already_added,duration_best_video)
    
    if video_path_file in dict_list_video_win:
        for other_video_path_file in dict_list_video_win[video_path_file]:
            generate_merge_command_other_part(other_video_path_file,dict_list_video_win,dict_file_path_obj,ffmpeg_cmd_dict,delay_to_put,common_language_use_for_generate_delay,md5_audio_already_added,md5_sub_already_added,duration_best_video)

def generate_new_file_audio_config(base_cmd,audio,md5_audio_already_added,audio_track_to_remove,delay_to_put):
    if ((not audio["keep"]) or (audio["MD5"] != '' and audio["MD5"] in md5_audio_already_added)):
        audio_track_to_remove.append(audio)
        return 0
    else:
        md5_audio_already_added.add(audio["MD5"])
        if audio["Format"].lower() == "flac" or ("Compression_Mode" in audio and audio["Compression_Mode"] == "Lossless"):
            if '@typeorder' in audio:
                base_cmd.extend([f"-c:a:{int(audio['@typeorder'])-1}", "flac", "-compression_level", "12"])
            else:
                base_cmd.extend([f"-c:a:0", "flac", "-compression_level", "12"])
            if "BitDepth" in audio:
                if audio["BitDepth"] == "16":
                    base_cmd.extend(["-sample_fmt", "s16"])
                else:
                    base_cmd.extend(["-sample_fmt", "s32"])
            else:
                base_cmd.extend(["-sample_fmt", "s32"])
            base_cmd.extend(["-exact_rice_parameters", "1"])
        elif delay_to_put < 0:
            if '@typeorder' in audio:
                base_cmd.extend([f"-c:a:{int(audio['@typeorder'])-1}"])
            else:
                base_cmd.extend([f"-c:a:0"])
            base_cmd.extend([audio["ffprobe"]["codec_name"]])
            try:
                if '@typeorder' in audio:
                    base_cmd.extend([f"-b:a:{int(audio['@typeorder'])-1}", video.get_bitrate(audio)])
                else:
                    base_cmd.extend([f"-b:a:0", video.get_bitrate(audio)])
            except:
                pass
        return 1

def generate_new_file(video_obj,delay_to_put,ffmpeg_cmd_dict,md5_audio_already_added,md5_sub_already_added,duration_best_video):
    base_cmd = [tools.software["ffmpeg"], "-err_detect", "crccheck", "-err_detect", "bitstream",
                    "-err_detect", "buffer", "-err_detect", "explode", "-max_muxing_queue_size", "8192",
                    "-probesize", "50000000",
                    "-threads", str(tools.core_to_use), "-vn"]
    if delay_to_put > 0:
        base_cmd.extend(["-itsoffset", f"{delay_to_put/Decimal(1000)}", "-i", video_obj.filePath])
    elif delay_to_put < 0:
        base_cmd.extend(["-i", video_obj.filePath, "-ss", f"{delay_to_put/Decimal(1000)*Decimal(-1)}"])
    else:
        base_cmd.extend(["-i", video_obj.filePath])

    base_cmd.extend(["-map", "0:a?", "-map", "0:s?", "-map_metadata", "0", "-copy_unknown",
                     "-movflags", "use_metadata_tags", "-c", "copy"])
    
    number_track = 0
    sub_track_to_remove = []
    for language,subs in video_obj.subtitles.items():
        if language in tools.language_to_completely_remove:
            for sub in subs:
                sub_track_to_remove.append(sub)
        else:
            for sub in subs:
                if (sub['keep'] and sub['MD5'] not in md5_sub_already_added):
                    number_track += 1
                    if sub['MD5'] != '':
                        md5_sub_already_added.add(sub['MD5'])
                else:
                    sub_track_to_remove.append(sub)
    
    audio_track_to_remove = []
    for language,audios in video_obj.audios.items():
        if language in tools.language_to_completely_remove:
            for audio in audios:
                audio_track_to_remove.append(audio)
        else:
            for audio in audios:
                number_track += generate_new_file_audio_config(base_cmd,audio,md5_audio_already_added,audio_track_to_remove,delay_to_put)
    for language,audios in video_obj.commentary.items():
        if language in tools.language_to_completely_remove:
            for audio in audios:
                audio_track_to_remove.append(audio)
        else:
            for audio in audios:
                number_track += generate_new_file_audio_config(base_cmd,audio,md5_audio_already_added,audio_track_to_remove,delay_to_put)
    for language,audios in video_obj.audiodesc.items():
        if language in tools.language_to_completely_remove:
            for audio in audios:
                audio_track_to_remove.append(audio)
        else:
            for audio in audios:
                number_track += generate_new_file_audio_config(base_cmd,audio,md5_audio_already_added,audio_track_to_remove,delay_to_put)
    
    if number_track:
        for audio in audio_track_to_remove:
            base_cmd.extend(["-map", f"-0:{audio["StreamOrder"]}"])
            
        for sub in sub_track_to_remove:
            base_cmd.extend(["-map", f"-0:{sub["StreamOrder"]}"])

        tmp_file_audio = path.join(tools.tmpFolder,f"{video_obj.fileBaseName}_tmp.mkv")
        base_cmd.extend(["-strict", "-2", "-t", duration_best_video, tmp_file_audio])

        ffmpeg_cmd_dict['convert_process'].append(video.ffmpeg_pool_audio_convert.apply_async(tools.launch_cmdExt, (base_cmd,)))
        ffmpeg_cmd_dict['merge_cmd'].extend(["--no-global-tags", "-M", "-B", tmp_file_audio])
    
    return number_track

def generate_launch_merge_command(dict_with_video_quality_logic,dict_file_path_obj,out_folder,common_language_use_for_generate_delay,audioRules):
    set_bad_video = set()
    dict_list_video_win = {}
    for video_path_file, dict_with_results in dict_with_video_quality_logic.items():
        for other_video_path_file, is_the_best_video in dict_with_results.items():
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
    if len(dict_list_video_win) == 0:
        dict_list_video_win[best_video.filePath] = []
    print(f'The best video path is {best_video.filePath}')
    md5_audio_already_added = set()
    md5_sub_already_added = set()
    
    ffmpeg_cmd_dict = {'files_with_offset' : [],
                       'number_files_add' : 0,
                       'convert_process' : [],
                       'merge_cmd' : [],
                       'metadata_cmd' : []}
    
    generate_new_file(best_video,0.0,ffmpeg_cmd_dict,md5_audio_already_added,md5_sub_already_added,best_video.video['Duration'])
    
    for video_obj_common_md5 in best_video.sameAudioMD5UseForCalculation:
        generate_merge_command_common_md5(video_obj_common_md5,0.0,ffmpeg_cmd_dict,md5_audio_already_added,md5_sub_already_added,best_video.video['Duration'])
    
    for other_video_path_file in dict_list_video_win[best_video.filePath]:
        generate_merge_command_other_part(other_video_path_file,dict_list_video_win,dict_file_path_obj,ffmpeg_cmd_dict,best_video.delays[common_language_use_for_generate_delay],common_language_use_for_generate_delay,md5_audio_already_added,md5_sub_already_added,best_video.video['Duration'])

    out_path_tmp_file_name_split = path.join(tools.tmpFolder,f"{best_video.fileBaseName}_merged_split.mkv")
    merge_cmd = [tools.software["mkvmerge"], "-o", out_path_tmp_file_name_split]
    merge_cmd.extend(ffmpeg_cmd_dict['merge_cmd'])
    for convert_process in ffmpeg_cmd_dict['convert_process']:
        convert_process.get()
    try:
        tools.launch_cmdExt(merge_cmd)
    except Exception as e:
        import re
        lined_error = str(e).splitlines()
        if re.match('Return code: 1', lined_error[-1]) != None:
            only_UID_warning = True
            i = 0
            while only_UID_warning and i < len(lined_error):
                if re.match('^Warning:.*', lined_error[i]) != None:
                    if re.match(r"^Warning:.+Could not keep a track's UID \d+ because it is already allocated for another track. A new random UID will be allocated automatically.", lined_error[i]) == None:
                        only_UID_warning = False
                i += 1
            if (not only_UID_warning):
                raise e
            else:
                sys.stderr.write(str(e))
        else:
            raise e

    tools.launch_cmdExt([tools.software["ffmpeg"], "-err_detect", "crccheck", "-err_detect", "bitstream",
                         "-err_detect", "buffer", "-err_detect", "explode", "-threads", str(tools.core_to_use),
                         "-i", out_path_tmp_file_name_split, "-map", "0", "-f", "null", "-c", "copy", "-"])
    
    out_video_metadata = video.video(tools.tmpFolder,path.basename(out_path_tmp_file_name_split))
    out_video_metadata.get_mediadata()
    out_video_metadata.video = best_video.video
    out_video_metadata.calculate_md5_streams_split()

    out_path_file_name = path.join(out_folder,f"{best_video.fileBaseName}_merged")
    if path.exists(out_path_file_name+'.mkv'):
        i = 1
        while path.exists(out_path_file_name+f'_({str(i)}).mkv'):
            i += 1
        out_path_file_name += f'_({str(i)}).mkv'
    else:
        out_path_file_name += '.mkv'
    
    final_insert = [tools.software["mkvmerge"], "-o", out_path_file_name]
    if tools.special_params["change_all_und"] and 'Language' not in best_video.video:
        final_insert.extend(["--language", best_video.video["StreamOrder"]+":"+tools.default_language_for_undetermine])
    final_insert.extend(["-A", "-S", "--no-chapters", best_video.filePath])
    
    list_track_order=[]
    global default_audio
    default_audio = True
    keep_best_audio(out_video_metadata.audios[common_language_use_for_generate_delay],audioRules)
    for audio_language in out_video_metadata.audios.keys():
        if audio_language != common_language_use_for_generate_delay:
            find_differences_and_keep_best_audio(out_video_metadata,audio_language,audioRules)

    number_track_audio = generate_merge_command_insert_ID_audio_track_to_remove_and_new_und_language(final_insert,out_video_metadata.audios,out_video_metadata.commentary,out_video_metadata.audiodesc,set(),list_track_order)
    
    sub_same_md5 = {}
    keep_sub = {'ass':[],'srt':[]}
    for language,subs in out_video_metadata.subtitles.items():
        for sub in subs:
            if sub['MD5'] in sub_same_md5:
                sub_same_md5[sub['MD5']].append(sub)
            else:
                sub_same_md5[sub['MD5']] = [sub]
    for sub_md5,subs in sub_same_md5.items():
        codec = sub['ffprobe']["codec_name"].lower()
        if len(subs) > 1:
            have_srt_sub = False
            for sub in subs:
                if sub['Format'].lower() in tools.sub_type_near_srt and (not have_srt_sub):
                    have_srt_sub = True
                    keep_sub["srt"].append(sub)
                else:
                    sub['keep'] = False
            if (not have_srt_sub):
                subs[0]['keep'] = True
                if codec not in tools.sub_type_not_encodable:
                    keep_sub["ass"].append(sub)
        else:
            if sub['Format'].lower() in tools.sub_type_near_srt:
                keep_sub["srt"].append(sub)
            elif codec not in tools.sub_type_not_encodable:
                keep_sub["ass"].append(sub)
    
    if len(keep_sub["srt"]) and len(keep_sub["ass"]):
        not_keep_ass_converted_in_srt(out_path_tmp_file_name_split,keep_sub["ass"],keep_sub["srt"])

    clean_number_stream_to_be_lover_than_max(max_stream-1-number_track_audio,out_video_metadata.subtitles)

    generate_merge_command_insert_ID_sub_track_set_not_default(final_insert,out_video_metadata.subtitles,set(),list_track_order)
    final_insert.extend(["-D", out_path_tmp_file_name_split])
    final_insert.extend(ffmpeg_cmd_dict['metadata_cmd'])
    final_insert.extend(["--track-order", f"0:{best_video.video["StreamOrder"]},1:"+",1:".join(list_track_order)])
    tools.launch_cmdExt(final_insert)
     
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
                audio["keep"] = (not tools.special_params["remove_commentary"])
        
    generate_launch_merge_command(dict_with_video_quality_logic,dict_file_path_obj,out_folder,"und",audioRules)
    
def sync_merge_video(videosObj,audioRules,out_folder,dict_file_path_obj,forced_best_video):
    commonLanguages = video.get_common_audios_language(videosObj)
    try:
        commonLanguages.remove("und")
    except:
        pass
    if len(commonLanguages) == 0:
        audio_counts = {}
        for videoObj in videosObj:
            for language in videoObj.audios.keys():
                if language not in audio_counts:
                    audio_counts[language] = 0
                audio_counts[language] += 1

        most_frequent_language = max(audio_counts, key=audio_counts.get)
        if audio_counts[most_frequent_language] == 1:
            raise Exception("No common language between "+str([videoObj.filePath for videoObj in videosObj]))
        else:
            commonLanguages.add(most_frequent_language)
            from sys import stderr
            i = 0
            list_video_not_compatible = []
            list_video_not_compatible_name = []
            for videoObj in videosObj:
                if most_frequent_language not in videoObj.audios:
                    list_video_not_compatible.append(i)
                    list_video_not_compatible_name.append(videoObj.filePath)
                i += 1
            for i in list_video_not_compatible:
                del videosObj[i]
            for i in list_video_not_compatible_name:
                del dict_file_path_obj[i]
            stderr.write(f"{[not_compatible_video for not_compatible_video in list_video_not_compatible_name]} not have the language {most_frequent_language}")
            stderr.write("\n")
    
    
    if len(commonLanguages) > 1 and tools.special_params["original_language"] in commonLanguages:
        common_language_use_for_generate_delay = tools.special_params["original_language"]
        commonLanguages.remove(common_language_use_for_generate_delay)
    else:
        commonLanguages = list(commonLanguages)
        common_language_use_for_generate_delay = commonLanguages.pop()
    
    MD5AudioVideo = {}
    listVideoToNotCalculateOffset = []
    for videoObj in videosObj:
        MD5merged = "".join(set([audio['MD5'] for audio in videoObj.audios[common_language_use_for_generate_delay]]))
        if MD5merged in MD5AudioVideo:
            if forced_best_video == videoObj.filePath:
                videoObj.sameAudioMD5UseForCalculation.append(MD5AudioVideo[MD5merged])
                listVideoToNotCalculateOffset.append(MD5AudioVideo[MD5merged])
                MD5AudioVideo[MD5merged] = videoObj
            else:
                MD5AudioVideo[MD5merged].sameAudioMD5UseForCalculation.append(videoObj)
                listVideoToNotCalculateOffset.append(videoObj)
        else:
            MD5AudioVideo[MD5merged] = videoObj
    
    for videoObj in listVideoToNotCalculateOffset:
        videosObj.remove(videoObj)
        del dict_file_path_obj[videoObj.filePath]
    
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
    
    generate_launch_merge_command(dict_with_video_quality_logic,dict_file_path_obj,out_folder,common_language_use_for_generate_delay,audioRules)
    
def merge_videos(files,out_folder,merge_sync,inFolder=None):
    videosObj = []
    name_file = {}
    files = list(files)
    files.sort()
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
    
    audioRules = decript_merge_rules(tools.mergeRules['audio'])
    
    dict_file_path_obj = {}
    forced_best_video = None
    md5_threads = []
    for videoObj in videosObj:
        process_mediadata_thread = Thread(target=videoObj.get_mediadata)
        process_mediadata_thread.start()
        dict_file_path_obj[videoObj.filePath] = videoObj
        if tools.special_params["forced_best_video"] != "":
            if tools.special_params["forced_best_video_contain"]:
                if tools.special_params["forced_best_video"] in videoObj.fileName:
                    forced_best_video = videoObj.filePath
            elif videoObj.fileName  == tools.special_params["forced_best_video"] or videoObj.filePath == tools.special_params["forced_best_video"]:
                forced_best_video = videoObj.filePath
        process_mediadata_thread.join()
        process_md5_thread = Thread(target=videoObj.calculate_md5_streams)
        process_md5_thread.start()
        md5_threads.append(process_md5_thread)
        
    for process_md5_thread in md5_threads:
        process_md5_thread.join()
    
    if merge_sync:
        sync_merge_video(videosObj,audioRules,out_folder,dict_file_path_obj,forced_best_video)
    else:
        simple_merge_video(videosObj,audioRules,out_folder,dict_file_path_obj,forced_best_video)