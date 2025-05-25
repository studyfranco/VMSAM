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
from threading import Thread
import tools
import video
from audioCorrelation import correlate, test_calcul_can_be, second_correlation
import json
from decimal import *
import math # Added for math.round

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
            return None
    
        calculated_delay = mean_between_delay+round(delay_second_method*1000) #delay_first_method+round(delay_second_method*1000)
        if abs(delay_second_method) < 0.125:
            # calculated_delay-delay_first_method_lower_result < 125 and calculated_delay-delay_first_method_lower_result > 0:
            sys.stderr.write(f"The delay {calculated_delay} find with adjuster_chroma_bugged is valid for {self.video_obj_1.filePath} and {self.video_obj_2.filePath}. The original delay was between {delay_first_method_lower_result} and {delay_first_method_bigger_result} \n")
            return calculated_delay
        else:
            sys.stderr.write(f"The delay {calculated_delay} find with adjuster_chroma_bugged is not valid for {self.video_obj_1.filePath} and {self.video_obj_2.filePath}. The original delay was between {delay_first_method_lower_result} and {delay_first_method_bigger_result} \n")
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

        score1 = self.video_obj_1.calculate_rational_score()
        score2 = self.video_obj_2.calculate_rational_score()

        rational_winner = 0 # 0 for tie, 1 for video_obj_1, 2 for video_obj_2
        # Compare first 5 elements (higher is better)
        for i in range(5):
            if score1[i] > score2[i]:
                rational_winner = 1
                break
            if score1[i] < score2[i]:
                rational_winner = 2
                break
        
        if rational_winner == 0: # If still tied, compare audio codec preference (lower is better)
            if score1[5] < score2[5]:
                rational_winner = 1
            elif score1[5] > score2[5]:
                rational_winner = 2

        # Main decision logic
        if rational_winner == 1 or \
           (rational_winner == 0 and video.get_best_quality_video(self.video_obj_1, self.video_obj_2, begins_video_for_compare_quality, self.time_by_test_best_quality_converted) == '1'):
            self.video_obj_1.extract_audio_in_part(self.language,self.audioParam,cutTime=self.list_cut_begin_length,asDefault=True)
            self.video_obj_2.remove_tmp_files(type_file="audio")
            self.video_obj_with_best_quality = self.video_obj_1
            delay = self.adjust_delay_to_frame(delay)
            self.video_obj_2.delays[self.language] += (delay*-Decimal(1.0)) # Delay you need to give to mkvmerge to be good.
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

def prepare_get_delay(videos_obj,language,audioRules):
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
                    if re.match(r".* *\[{0,1}forced\]{0,1} *.*", sub["Title"].lower()):
                        merge_cmd.extend(["--forced-display-flag", sub["StreamOrder"]+":1"])
                        if language_and_type+'_forced' not in dic_language_list_track_ID:
                            dic_language_list_track_ID[language_and_type+'_forced'] = [sub["StreamOrder"]]
                        else:
                            dic_language_list_track_ID[language_and_type+'_forced'].append(sub["StreamOrder"])
                    elif re.match(r".* *\[{0,1}sdh\]{0,1} *.*", sub["Title"].lower()):
                        merge_cmd.extend(["--hearing-impaired-flag", sub["StreamOrder"]+":1"])
                        if language_and_type+'_hearing' not in dic_language_list_track_ID:
                            dic_language_list_track_ID[language_and_type+'_hearing'] = [sub["StreamOrder"]]
                        else:
                            dic_language_list_track_ID[language_and_type+'_hearing'].append(sub["StreamOrder"])
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

def generate_new_file_audio_config(base_cmd, audio, md5_audio_already_added, audio_track_to_remove, delay_to_put_ms, duration_best_video_str):
    if ((not audio["keep"]) or (audio["MD5"] != '' and audio["MD5"] in md5_audio_already_added)):
        audio_track_to_remove.append(audio)
        return 0
    else:
        md5_audio_already_added.add(audio["MD5"])
        
        needs_padding = False
        padding_duration_seconds = 0.0
        
        if duration_best_video_str and delay_to_put_ms is not None: # Ensure params are available
            try:
                offset_seconds = float(delay_to_put_ms / Decimal(1000))
                best_video_duration_seconds = float(duration_best_video_str)
                current_audio_duration_seconds = float(audio['Duration'])
                current_audio_effective_end_time = offset_seconds + current_audio_duration_seconds
                
                if current_audio_effective_end_time < best_video_duration_seconds:
                    padding_duration_seconds = best_video_duration_seconds - current_audio_effective_end_time
                    if padding_duration_seconds > 0.001:
                        needs_padding = True
            except (ValueError, TypeError, KeyError) as e:
                sys.stderr.write(f"Error calculating padding for {audio.get('StreamOrder', 'N/A')}: {e}\n")
                # Continue without padding if there's an error in calculation
        
        stream_specifier_codec = f":a:{int(audio['@typeorder'])-1}" if '@typeorder' in audio else ":a:0"

        if needs_padding:
            base_cmd.extend([f"-af{stream_specifier_codec}", f"apad=pad_dur={padding_duration_seconds}"])
            if audio["Format"].lower() == "flac":
                base_cmd.extend([f"-c{stream_specifier_codec}", "flac", "-compression_level", "12"])
                if "BitDepth" in audio:
                    if audio["BitDepth"] == "16":
                        base_cmd.extend(["-sample_fmt", "s16"])
                    else:
                        base_cmd.extend(["-sample_fmt", "s32"])
                else: # Default to s16 if BitDepth is missing for FLAC
                    base_cmd.extend(["-sample_fmt", "s16"])
            else: # Not FLAC, re-encode to FLAC with padding
                base_cmd.extend([f"-c{stream_specifier_codec}", "flac", "-compression_level", "12", "-sample_fmt", "s16"])
        elif audio["Format"].lower() == "flac": # No padding, but original is FLAC, apply FLAC options
            base_cmd.extend([f"-c{stream_specifier_codec}", "flac", "-compression_level", "12"])
            if "BitDepth" in audio:
                if audio["BitDepth"] == "16":
                    base_cmd.extend(["-sample_fmt", "s16"])
                else:
                    base_cmd.extend(["-sample_fmt", "s32"])
            else: # Default to s16 if BitDepth is missing for FLAC
                 base_cmd.extend(["-sample_fmt", "s16"])
        # If not padding and not FLAC, it will be handled by global '-c copy' or other specific rules.
        return 1

def generate_new_file(video_obj,delay_to_put,ffmpeg_cmd_dict,md5_audio_already_added,md5_sub_already_added,duration_best_video):
    # delay_to_put is Decimal in milliseconds. duration_best_video is string "seconds.milliseconds"
    base_cmd = [tools.software["ffmpeg"], "-err_detect", "crccheck", "-err_detect", "bitstream",
                    "-err_detect", "buffer", "-err_detect", "explode", "-fflags", "+genpts+igndts",
                    "-threads", str(tools.core_to_use), "-vn"] # -vn must be early
    
    # Input file and its offset processing must come before output options and mappings
    # Create a preliminary command list for input processing
    input_cmd_part = []
    if delay_to_put != 0:
        input_cmd_part.extend(["-itsoffset", f"{delay_to_put/Decimal(1000)}"])
    input_cmd_part.extend(["-i", video_obj.filePath])

    # Global output options that might be overridden by stream-specific ones later
    output_options_part = ["-map", "0:a?", "-map", "0:s?", "-map_metadata", "0", "-copy_unknown",
                           "-movflags", "use_metadata_tags", "-c", "copy", "-c:s", "ass"]
    
    # Combine parts: ffmpeg base, input processing, then global output options
    # Stream specific options will be added by generate_new_file_audio_config
    base_cmd.extend(input_cmd_part)
    # Audio config (potentially adding filters and codec changes) must be defined *before* output_options_part if those options are to take effect
    # However, ffmpeg command structure usually is: ffmpeg [global_opts] [input_opts] -i input [output_opts] output
    # Stream specific options like -c:a:0, -af:a:0 are output options.
    # So, the current structure of appending to base_cmd in generate_new_file_audio_config should be okay,
    # as long as base_cmd is then extended with output_options_part *after* audio processing is determined.
    # This means generate_new_file_audio_config should modify a part of command that is later inserted.
    # For now, let's stick to current append model and test. If issues, restructure base_cmd assembly.
    
    number_track = 0
    sub_track_to_remove = []
    # Collect subtitle processing commands
    subtitle_codec_cmds = []
    for language,subs in video_obj.subtitles.items():
        for sub in subs:
            if (sub['keep'] and sub['MD5'] not in md5_sub_already_added):
                number_track += 1
                if sub['MD5'] != '':
                    md5_sub_already_added.add(sub['MD5'])
                codec = sub["Format"].lower()
                stream_specifier_sub = f":s:{int(sub['@typeorder'])-1}" if '@typeorder' in sub else ":s:0" # Assuming @typeorder for subs too
                if codec in tools.sub_type_not_encodable:
                    subtitle_codec_cmds.extend([f"-c{stream_specifier_sub}", "copy"])
                elif codec in tools.sub_type_near_srt:
                     subtitle_codec_cmds.extend([f"-c{stream_specifier_sub}", "srt"])
                # else, it will be converted to ass by global -c:s ass
            else:
                sub_track_to_remove.append(sub)
    
    # Collect audio processing commands (including padding/re-encoding)
    audio_processing_cmds = []
    audio_track_to_remove = []
    for language,audios in video_obj.audios.items():
        for audio in audios:
            number_track += generate_new_file_audio_config(audio_processing_cmds, audio, md5_audio_already_added, audio_track_to_remove, delay_to_put, duration_best_video)
    for language,audios in video_obj.commentary.items():
        for audio in audios:
            number_track += generate_new_file_audio_config(audio_processing_cmds, audio, md5_audio_already_added, audio_track_to_remove, delay_to_put, duration_best_video)
    for language,audios in video_obj.audiodesc.items():
        for audio in audios:
            number_track += generate_new_file_audio_config(audio_processing_cmds, audio, md5_audio_already_added, audio_track_to_remove, delay_to_put, duration_best_video)

    # Now assemble the full command
    # base_cmd already has ffmpeg, global error flags, -vn
    # then add input processing
    base_cmd.extend(input_cmd_part)
    # then add global output options
    base_cmd.extend(output_options_part)
    # then add specific subtitle codec commands
    base_cmd.extend(subtitle_codec_cmds)
    # then add specific audio processing commands (filters, codecs)
    base_cmd.extend(audio_processing_cmds)
    
    if number_track:
        # Add track removal maps
        for audio in audio_track_to_remove:
            base_cmd.extend(["-map", f"-0:{audio["StreamOrder"]}"])
        for sub in sub_track_to_remove:
            base_cmd.extend(["-map", f"-0:{sub["StreamOrder"]}"])

        tmp_file_audio = path.join(tools.tmpFolder,f"{video_obj.fileBaseName}_tmp.mkv")
        base_cmd.extend(["-t", duration_best_video, tmp_file_audio]) # Output path must be last before options for it
        
        current_ffmpeg_cmd = base_cmd # Create a copy if base_cmd is reused or modified later for other files
        ffmpeg_cmd_dict['convert_process'].append(video.ffmpeg_pool_audio_convert.apply_async(tools.launch_cmdExt, (current_ffmpeg_cmd,)))
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
    
    # --- Subtitle Deduplication and Preference Logic (Retained) ---
    sub_same_md5 = {}
    # Populate sub_same_md5 correctly first
    for lang_key, subs_list_for_lang in out_video_metadata.subtitles.items():
        for sub_item in subs_list_for_lang:
            if sub_item['MD5'] in sub_same_md5:
                sub_same_md5[sub_item['MD5']].append(sub_item)
            else:
                sub_same_md5[sub_item['MD5']] = [sub_item]

    keep_sub = {'ass':[],'srt':[]} # Stores references to tracks to keep for not_keep_ass_converted_in_srt
    for sub_md5, list_of_subs_with_same_md5 in sub_same_md5.items():
        if not list_of_subs_with_same_md5:
            continue
        
        codec = "unknown" 
        if 'ffprobe' in list_of_subs_with_same_md5[0] and 'codec_name' in list_of_subs_with_same_md5[0]['ffprobe']:
             codec = list_of_subs_with_same_md5[0]['ffprobe']["codec_name"].lower()
        
        if len(list_of_subs_with_same_md5) > 1: # Multiple tracks with the same MD5
            have_srt_sub = False
            for s_track in list_of_subs_with_same_md5: 
                if s_track['Format'].lower() in tools.sub_type_near_srt and not have_srt_sub:
                    have_srt_sub = True
                    keep_sub["srt"].append(s_track)
                    s_track['keep'] = True 
                else:
                    s_track['keep'] = False 
            if not have_srt_sub: # No SRT found among duplicates, keep the first one (could be ASS, PGS etc.)
                list_of_subs_with_same_md5[0]['keep'] = True
                if codec not in tools.sub_type_not_encodable and codec not in tools.sub_type_near_srt : 
                    keep_sub["ass"].append(list_of_subs_with_same_md5[0])
        else: # Only one sub with this MD5
            s_track = list_of_subs_with_same_md5[0]
            s_track['keep'] = True # Keep it by default (subject to SRT vs ASS conversion check)
            if s_track['Format'].lower() in tools.sub_type_near_srt:
                keep_sub["srt"].append(s_track)
            elif codec not in tools.sub_type_not_encodable: 
                keep_sub["ass"].append(s_track)
    
    if len(keep_sub["srt"]) and len(keep_sub["ass"]):
        not_keep_ass_converted_in_srt(out_path_tmp_file_name_split,keep_sub["ass"],keep_sub["srt"])
    
    # --- Blank Subtitle Generation Logic (Removed) ---
    # The complex loop iterating all_subtitle_languages, checking durations, 
    # and generating blank SRTs has been removed.
    # generate_chimeric_video is now responsible for ensuring full-duration subs.

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
    
    md5_audio_already_added_for_final = set()
    generate_merge_command_insert_ID_audio_track_to_remove_and_new_und_language(final_insert,out_video_metadata.audios,out_video_metadata.commentary,out_video_metadata.audiodesc,md5_audio_already_added_for_final,list_track_order)
    
    md5_sub_already_added_for_final = set() # Use a new set for subtitles for this final merge stage
    generate_merge_command_insert_ID_sub_track_set_not_default(final_insert,out_video_metadata.subtitles,md5_sub_already_added_for_final,list_track_order)
    
    # Removed: Loop for blank_sub_files_to_add, as this is handled earlier.

    final_insert.extend(["-D", out_path_tmp_file_name_split]) 
    final_insert.extend(ffmpeg_cmd_dict['metadata_cmd']) 
    
    # Final track order: best video first, then audio, then subtitles (including blanks implicitly at end of subtitle group)
    if list_track_order: # Ensure list_track_order is not empty
        final_insert.extend(["--track-order", f"0:{best_video.video['StreamOrder']},1:"+",1:".join(list_track_order)])
    else: # Fallback if no audio/subs were explicitly ordered (e.g. only video track)
        final_insert.extend(["--track-order", f"0:{best_video.video['StreamOrder']}"])
        
    tools.launch_cmdExt(final_insert)

def seconds_to_srt_timeformat(seconds_float):
    hours = int(seconds_float // 3600)
    seconds_float %= 3600
    minutes = int(seconds_float // 60)
    seconds_float %= 60
    seconds = int(seconds_float)
    milliseconds = int(round((seconds_float - seconds) * 1000))
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

import os # For os.remove in cleanup
import math # For math.floor, math.ceil if needed, or just general math.

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
    
    original_video_obj_count = len(videosObj)
    for videoObj_to_remove in listVideoToNotCalculateOffset: # Iterate over a copy
        if videoObj_to_remove in videosObj: # Check if still in list before removing
            videosObj.remove(videoObj_to_remove)
        if videoObj_to_remove.filePath in dict_file_path_obj:
            del dict_file_path_obj[videoObj_to_remove.filePath]
    
    # Ensure dict_file_path_obj is consistent with videosObj after removals
    current_file_paths_in_videosObj = {v.filePath for v in videosObj}
    for path_key in list(dict_file_path_obj.keys()): # Iterate over a copy of keys
        if path_key not in current_file_paths_in_videosObj:
            del dict_file_path_obj[path_key]

    if not videosObj: # All videos were duplicates based on MD5 audio for sync language
        sys.stderr.write("Error: No unique videos left after MD5 audio check for sync language. Cannot proceed.\n")
        if original_video_obj_count > 0 and listVideoToNotCalculateOffset: # This means all videos were duplicates of the first unique one
             # Re-add the first unique video (representative) to proceed if needed, or handle error.
             # For now, error out if empty.
             return

    dict_with_video_quality_logic = {}
    if forced_best_video == None:
        if not videosObj: # Should be caught above, but double check.
             sys.stderr.write("Error: videosObj is empty before calling get_delay_and_best_video.\n")
             return
        dict_with_video_quality_logic = get_delay_and_best_video(videosObj,common_language_use_for_generate_delay,audioRules,dict_file_path_obj)
    else:
        print_forced_video(forced_best_video)
        if forced_best_video not in dict_file_path_obj:
            sys.stderr.write(f"Error: Forced best video {forced_best_video} not found in available video objects. Cannot proceed.\n")
            # This could happen if the forced best video was among the MD5 duplicates removed.
            # Need to ensure forced_best_video is retained if it was part of an MD5 duplicate set and was the chosen one.
            # The MD5 logic might need adjustment to always keep 'forced_best_video' if it's set.
            # For now, assuming it's an error if not found.
            return
        dict_with_video_quality_logic = get_delay(videosObj,common_language_use_for_generate_delay,audioRules,dict_file_path_obj,forced_best_video)

    # --- V_abs_best Identification ---
    set_bad_video_paths = set()
    for video_path_file_iter, dict_with_results_iter in dict_with_video_quality_logic.items():
        for other_video_path_file_iter, is_the_best_video_iter in dict_with_results_iter.items():
            if is_the_best_video_iter:
                set_bad_video_paths.add(other_video_path_file_iter)
            elif is_the_best_video_iter is False: # Explicitly False, None means not compared or error
                set_bad_video_paths.add(video_path_file_iter)
            # If is_the_best_video_iter is None, it means they were incompatible or not compared.
            # This logic might need to handle None from already_compared if that's possible.
            # Current get_delay_and_best_video returns already_compared which has None for incompatible.
            # If None means "other video is not worse", then current logic is okay.
            # If None means "this comparison was problematic", those videos might need removal or special handling.
            # Assuming for now that None values in dict_with_video_quality_logic mean those pairs are not considered for "best".

    V_abs_best = None
    possible_best_paths = list(set(dict_file_path_obj.keys()) - set_bad_video_paths)
    if not possible_best_paths:
        if len(dict_file_path_obj) == 1:
            V_abs_best = list(dict_file_path_obj.values())[0]
        elif not dict_file_path_obj:
             sys.stderr.write("ERROR: No videos available to determine V_abs_best.\n")
             return
        else: # Multiple videos, all marked bad or no clear winner
            sys.stderr.write("ERROR: No best video found after initial comparisons, or all videos marked as bad.\n")
            # Fallback: If forced_best_video is set and still in dict_file_path_obj, use it.
            if forced_best_video and forced_best_video in dict_file_path_obj:
                V_abs_best = dict_file_path_obj[forced_best_video]
                sys.stderr.write(f"Warning: Using forced best video {V_abs_best.filePath} as V_abs_best due to lack of clear winner.\n")
            else: # Truly no idea what is best.
                 return # Cannot proceed.
    elif len(possible_best_paths) > 1:
        sys.stderr.write(f"Warning: Multiple best videos identified: {possible_best_paths}. Using the first one: {possible_best_paths[0]}\n")
        V_abs_best = dict_file_path_obj[possible_best_paths[0]]
    else: # Exactly one best path
        V_abs_best = dict_file_path_obj[possible_best_paths[0]]

    if V_abs_best is None :
        sys.stderr.write("Critical Error: V_abs_best could not be determined. Aborting sync_merge_video.\n")
        return

    print(f"Absolute best video identified as: {V_abs_best.filePath}")

    # --- Chimeric Video Generation ---
    audio_params_for_wav = {'Format':"WAV", 'codec':"pcm_s16le", 'Channels': "2"} # Defaulting to 2 channels for intermediate WAVs
    
    updated_dict_file_path_obj = {V_abs_best.filePath: V_abs_best}
    paths_to_remove_post_chimeric = []
    # Map original paths to their new chimeric paths if successful
    original_to_chimeric_map = {} 

    for original_video_path, video_obj_to_conform in list(dict_file_path_obj.items()):
        if video_obj_to_conform.filePath == V_abs_best.filePath:
            continue

        # Calculate length_time_initial_segment_chunk_duration for this pair
        min_duration_for_pair_s = video.get_shortest_audio_durations([V_abs_best, video_obj_to_conform], common_language_use_for_generate_delay)
        if min_duration_for_pair_s == 0: # One video has no audio for this lang, or zero duration
            sys.stderr.write(f"Warning: Cannot determine audio duration for pair ({V_abs_best.filePath}, {video_obj_to_conform.filePath}) in lang {common_language_use_for_generate_delay}. Skipping chimeric.\n")
            paths_to_remove_post_chimeric.append(original_video_path)
            continue
            
        _unused_begin_s, length_time_for_pair_chunk_s = video.generate_begin_and_length_by_segment(min_duration_for_pair_s)
        if length_time_for_pair_chunk_s <=0 :
            sys.stderr.write(f"Warning: Calculated zero or negative chunk duration for pair ({V_abs_best.filePath}, {video_obj_to_conform.filePath}). Skipping chimeric.\n")
            paths_to_remove_post_chimeric.append(original_video_path)
            continue

        print(f"Conforming {video_obj_to_conform.filePath} to {V_abs_best.filePath}...")
        chimeric_mkv_path = generate_chimeric_video(
            V_abs_best, video_obj_to_conform, 
            common_language_use_for_generate_delay, 
            audio_params_for_wav, # For segment-wise correlation in generate_chimeric_video
            length_time_for_pair_chunk_s, # For initial offset calculation within generate_chimeric_video
            tools.tmpFolder, tools.tmpFolder # Chimeric files created in tmpFolder
        )

        if chimeric_mkv_path is None:
            print(f"Failed to generate chimeric video for {video_obj_to_conform.filePath}.")
            paths_to_remove_post_chimeric.append(original_video_path)
        else:
            print(f"Generated chimeric video: {chimeric_mkv_path}")
            chimeric_video = video.video(tools.tmpFolder, path.basename(chimeric_mkv_path))
            chimeric_video.get_mediadata()
            chimeric_video.calculate_md5_streams_split() # Important for generate_launch_merge_command
            chimeric_video.delays[common_language_use_for_generate_delay] = Decimal(0) # Conformed
            updated_dict_file_path_obj[chimeric_video.filePath] = chimeric_video
            original_to_chimeric_map[original_video_path] = chimeric_video.filePath
            # The original video_obj_to_conform (and its temp files) can be cleaned up if no longer needed
            # For now, it's just removed from the list for final merge.
            # If original_video_path != chimeric_mkv_path, means a new file was created.
            # The original can be marked for deletion from disk later if desired.
            if original_video_path in dict_file_path_obj: # Should always be true here
                 if original_video_path != chimeric_video.filePath: # If new file created, old one is no longer primary
                     paths_to_remove_post_chimeric.append(original_video_path)


    dict_file_path_obj = updated_dict_file_path_obj

    for path_to_remove in paths_to_remove_post_chimeric:
        if path_to_remove in dict_file_path_obj:
            del dict_file_path_obj[path_to_remove]
        # Also remove from original_to_chimeric_map if it was a key, though it's used to build the new dict.
        # This ensures only successfully conformed or original best video is in dict_file_path_obj

    # Rebuild dict_with_video_quality_logic for generate_launch_merge_command
    # It now reflects V_abs_best vs successfully created (and kept) chimeric videos.
    # All chimeric videos are "worse" than V_abs_best in terms of being the reference.
    new_dict_with_video_quality_logic = {}
    if V_abs_best.filePath in dict_file_path_obj: # V_abs_best should always be here
        new_dict_with_video_quality_logic[V_abs_best.filePath] = {}
        for video_fpath_key in dict_file_path_obj.keys():
            if video_fpath_key != V_abs_best.filePath:
                new_dict_with_video_quality_logic[V_abs_best.filePath][video_fpath_key] = True # True means V_abs_best is "better" (the reference)

    dict_with_video_quality_logic = new_dict_with_video_quality_logic
    
    # At this point, dict_file_path_obj contains V_abs_best and any successfully created chimeric videos.
    # All objects in dict_file_path_obj should have delays relative to V_abs_best (0 for chimeras and V_abs_best itself).

    for language in commonLanguages: # Other common languages, if any.
        # This loop was for cross-validation, which is not part of chimeric generation directly.
        # If needed, delays for other languages in chimeric videos would also be 0 relative to V_abs_best's version of that lang.
        pass
    
    if not dict_file_path_obj:
        sys.stderr.write("No videos remaining after chimeric processing. Aborting.\n")
        return

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

# Helper function for formatting time to HH:MM:SS.mmm
def format_time_ffmpeg(time_s): # Already exists from previous step
    ms = int((time_s - int(time_s)) * 1000)
    return f"{strftime('%H:%M:%S', gmtime(int(time_s)))}.{ms:03d}"

# Helper function for cleaning up temporary files (generic now)
def cleanup_temp_files(file_list):
    for f_path in file_list:
        if os.path.exists(f_path):
            try:
                os.remove(f_path)
            except OSError as e:
                sys.stderr.write(f"Error removing temp file {f_path}: {e}\n")

# Renamed from cleanup_video_obj_temp_audio for clarity, as it's for segment files now primarily
def cleanup_video_obj_temp_audio(video_objs_list):
    for video_obj in video_objs_list:
        if video_obj and 'audio' in video_obj.tmpFiles and video_obj.tmpFiles['audio']:
            try:
                video_obj.remove_tmp_files(type_file="audio") # This cleans temp files managed by extract_audio_in_part
            except Exception as e:
                sys.stderr.write(f"Error cleaning up video_obj internal temp audio for {video_obj.filePath}: {e}\n")

# Note: audio_param_for_extraction is for the main segment loop (WAV for correlation)
# For initial offset, generate_chimeric_video will use WAV params internally for get_delay_fidelity
def generate_chimeric_video(main_video_obj, other_video_obj, sync_language, audio_param_for_extraction, length_time_initial_segment_chunk_duration, tools_tmpFolder, out_folder_for_chimeric_file):
    # --- Initial Offset Calculation ---
    initial_offset_ms = None
    # Define audio params for initial offset calculation (must be WAV for get_delay_fidelity)
    initial_offset_audio_params = {'Format': "WAV", 'codec': "pcm_s16le", 'Channels': "2"} # Default to 2 channels for initial check
    # Adjust channels if necessary based on source videos for robustness, or ensure get_less_channel_number handles it.
    # For simplicity, using "2" as per existing prepare_get_delay, assuming get_less_channel_number would be called by a higher-level orchestrator if needed.

    # Determine a suitable duration for initial segments for offset calculation
    # length_time_initial_segment_chunk_duration is the duration of each small chunk for get_delay_fidelity
    # We need a total segment that's video.number_cut * this chunk duration.
    # Let's use a fixed number of cuts for initial offset or make it dependent on video.number_cut.
    # Using video.number_cut for consistency with how first_delay_test works.
    
    # Min duration between the two for determining segment lengths for initial offset
    # This logic should ideally be outside or passed in if it needs full video list context.
    # For now, calculate based on the pair.
    min_duration_pair_s = min(float(main_video_obj.video['Duration']), float(other_video_obj.video['Duration']))
    
    # begin_s_for_initial_offset is the start of the overall segment from which chunks are taken. Typically 0 or near 0.
    # length_time_for_chunk_calc is the duration of each chunk used by get_delay_fidelity.
    # This should be length_time_initial_segment_chunk_duration
    begin_s_for_initial_offset, _ = video.generate_begin_and_length_by_segment(min_duration_pair_s) # Use the chunk duration passed
                                                                                                     # No, this recalculates chunk_duration. We need to use the passed one.
                                                                                                     # Let's assume begin_s_for_initial_offset can be fixed (e.g. 0 or small value)
                                                                                                     # and length_time_initial_segment_chunk_duration is the chunk size.
    
    # Simplified: Use a fixed beginning (e.g. 0s or a small offset like 5s) for initial extraction.
    # The actual begin_in_second for generate_cut_with_begin_length for get_delay_fidelity
    # This is complex as generate_begin_and_length_by_segment determines both begin and chunk length
    # Let's assume length_time_initial_segment_chunk_duration is the chunk length (float, seconds)
    # And we'll use a fixed number of cuts (e.g., video.number_cut) starting from a small offset.
    
    initial_extraction_begin_offset_s = 5.0 # Start a bit into the video to avoid intros/blank screens
    if min_duration_pair_s < (initial_extraction_begin_offset_s + video.number_cut * length_time_initial_segment_chunk_duration):
        initial_extraction_begin_offset_s = 0.0 # Fallback to 0 if video too short for this offset
    
    if min_duration_pair_s < video.number_cut * length_time_initial_segment_chunk_duration:
         sys.stderr.write(f"Warning: Videos too short for full initial offset segment calculation. Pair: {main_video_obj.filePath}, {other_video_obj.filePath}\n")
         # Potentially adjust number_cut or chunk_duration if too short, or rely on get_delay_fidelity handling it.
         # For now, proceed, get_delay_fidelity might fail or give fewer points.


    initial_main_cuts = video.generate_cut_with_begin_length(initial_extraction_begin_offset_s, length_time_initial_segment_chunk_duration, format_time_ffmpeg(length_time_initial_segment_chunk_duration))
    initial_other_cuts = video.generate_cut_with_begin_length(initial_extraction_begin_offset_s, length_time_initial_segment_chunk_duration, format_time_ffmpeg(length_time_initial_segment_chunk_duration))

    main_video_obj.extract_audio_in_part(sync_language, initial_offset_audio_params, cutTime=initial_main_cuts, asDefault=True)
    other_video_obj.extract_audio_in_part(sync_language, initial_offset_audio_params, cutTime=initial_other_cuts, asDefault=True)
    
    delay_Fidelity_Values = get_delay_fidelity(main_video_obj, other_video_obj, length_time_initial_segment_chunk_duration) # Pass chunk duration
    
    stable_delays_ms = []
    if not delay_Fidelity_Values:
        sys.stderr.write(f"Initial offset calculation: get_delay_fidelity returned no results for {main_video_obj.filePath} and {other_video_obj.filePath}.\n")
        cleanup_video_obj_temp_audio([main_video_obj, other_video_obj])
        return None

    for key_audio_pair, delay_fidelity_list_for_pair in delay_Fidelity_Values.items():
        if not delay_fidelity_list_for_pair: continue
        # Check for consistent delay within this pair's segments
        first_offset_val = delay_fidelity_list_for_pair[0][2] 
        if all(abs(df[2] - first_offset_val) < 10 for df in delay_fidelity_list_for_pair): # Allow small tolerance
            stable_delays_ms.append(first_offset_val)
        else: # Inconsistent segment delays for this audio track pair
            sys.stderr.write(f"Initial offset: Inconsistent delays for audio pair {key_audio_pair} between {main_video_obj.filePath} and {other_video_obj.filePath}. Delays: {[df[2] for df in delay_fidelity_list_for_pair]}\n")
            # This might be an error condition depending on strictness. For now, we collect all stable delays.

    if not stable_delays_ms:
        sys.stderr.write(f"Initial offset calculation: No stable delays found for any audio track pair between {main_video_obj.filePath} and {other_video_obj.filePath}.\n")
        cleanup_video_obj_temp_audio([main_video_obj, other_video_obj])
        return None
    
    # Check consistency *across* different audio track pairs if multiple exist (e.g. main_audio0-other_audio0 vs main_audio1-other_audio1)
    # For now, using the first stable delay found. If multiple audio pairs yielded stable but different offsets, this might be an issue.
    initial_offset_ms = Decimal(str(stable_delays_ms[0])) 
    if len(set(stable_delays_ms)) > 1: # More than one distinct stable delay found
        # Check if they are very close. If not, it's problematic.
        if max(stable_delays_ms) - min(stable_delays_ms) > 50: # Example: 50ms tolerance for different audio pairs
             sys.stderr.write(f"Warning: Multiple different stable initial offsets found: {stable_delays_ms}. Using first: {initial_offset_ms}ms.\n")
        else: # All stable delays are close, average them or use first.
            initial_offset_ms = Decimal(str(round(sum(stable_delays_ms) / len(stable_delays_ms)))) # Average close delays
            sys.stdout.write(f"Info: Averaged close initial offsets to: {initial_offset_ms}ms.\n")


    sys.stdout.write(f"Initial offset calculated for {other_video_obj.filePath} relative to {main_video_obj.filePath}: {initial_offset_ms} ms\n")
    
    # Cleanup audio files used for initial offset calculation before main loop uses extract_audio_in_part
    cleanup_video_obj_temp_audio([main_video_obj, other_video_obj])


    FIDELITY_THRESHOLD = 0.80
    SEGMENT_DURATION_S = 60.0
    INCOMPATIBILITY_THRESHOLD_PERCENT = 50.0
    OFFSET_CONSISTENCY_WINDOW_S = 0.5

    # Ensure `tools_tmpFolder` is available (passed as argument)

    segment_assembly_plan = []
    current_main_time_s = 0.0
    # initial_offset_ms is now calculated above
    current_other_offset_s = float(initial_offset_ms / Decimal(1000))

    try:
        main_total_duration_s = float(main_video_obj.video['Duration'])
        other_total_duration_s = float(other_video_obj.video['Duration'])
    except (KeyError, ValueError, TypeError) as e:
        sys.stderr.write(f"Error getting video durations: {e}\n")
        # No specific cleanup here as initial offset files already cleaned
        return None

    total_bad_segments_duration_s = 0.0
    total_processed_duration_s = 0.0

    # --- Segment processing loop (largely as before) ---
    while current_main_time_s < main_total_duration_s:
        actual_segment_duration_s = min(SEGMENT_DURATION_S, main_total_duration_s - current_main_time_s)
        if actual_segment_duration_s < 1.0: break
        main_segment_start_str = format_time_ffmpeg(current_main_time_s)
        main_segment_duration_str = format_time_ffmpeg(actual_segment_duration_s)
        main_cut_time = [[main_segment_start_str, main_segment_duration_str]]
        try:
            # For the main loop, use the audio_param_for_extraction passed in (which is for correlation)
            main_video_obj.extract_audio_in_part(sync_language, audio_param_for_extraction, cutTime=main_cut_time, asDefault=True)
            if not main_video_obj.tmpFiles.get('audio') or not main_video_obj.tmpFiles['audio'][0]: raise Exception("Main audio seg path not found.")
            main_audio_segment_path = main_video_obj.tmpFiles['audio'][0][0]
        except Exception as e:
            sys.stderr.write(f"Err extracting main audio {main_segment_start_str}: {e}\n")
            segment_assembly_plan.append({'type': 'bad', 'main_start_s': current_main_time_s, 'duration_s': actual_segment_duration_s, 'reason': 'main_extract_fail'})
            total_bad_segments_duration_s += actual_segment_duration_s
            total_processed_duration_s += actual_segment_duration_s; current_main_time_s += actual_segment_duration_s; continue
        other_segment_start_abs_s = current_main_time_s - current_other_offset_s
        if (other_segment_start_abs_s + actual_segment_duration_s < 0) or (other_segment_start_abs_s >= other_total_duration_s):
            segment_assembly_plan.append({'type': 'bad', 'main_start_s': current_main_time_s, 'duration_s': actual_segment_duration_s, 'reason': 'other_out_of_bounds'})
            total_bad_segments_duration_s += actual_segment_duration_s
        else:
            other_segment_start_clipped_s = max(0.0, other_segment_start_abs_s)
            other_segment_duration_clipped_s = min(actual_segment_duration_s, other_total_duration_s - other_segment_start_clipped_s)
            if other_segment_duration_clipped_s < 1.0:
                segment_assembly_plan.append({'type': 'bad', 'main_start_s': current_main_time_s, 'duration_s': actual_segment_duration_s, 'reason': 'other_clip_too_short'})
                total_bad_segments_duration_s += actual_segment_duration_s
            else:
                other_segment_start_str = format_time_ffmpeg(other_segment_start_clipped_s); other_segment_duration_str = format_time_ffmpeg(other_segment_duration_clipped_s)
                other_cut_time = [[other_segment_start_str, other_segment_duration_str]]
                try:
                    other_video_obj.extract_audio_in_part(sync_language, audio_param_for_extraction, cutTime=other_cut_time, asDefault=True)
                    if not other_video_obj.tmpFiles.get('audio') or not other_video_obj.tmpFiles['audio'][0]: raise Exception("Other audio seg path not found.")
                    other_audio_segment_path = other_video_obj.tmpFiles['audio'][0][0]
                    fidelity, _, _ = correlate(main_audio_segment_path, other_audio_segment_path, other_segment_duration_clipped_s)
                    _shifted_file, precise_offset_s_fft_raw = second_correlation(main_audio_segment_path, other_audio_segment_path)
                    precise_offset_s_fft = -precise_offset_s_fft_raw if _shifted_file == main_audio_segment_path else precise_offset_s_fft_raw
                    new_potential_other_offset_s = current_other_offset_s - precise_offset_s_fft
                    offset_drift_s = abs(new_potential_other_offset_s - current_other_offset_s)
                    if fidelity >= FIDELITY_THRESHOLD and offset_drift_s <= OFFSET_CONSISTENCY_WINDOW_S:
                        current_other_offset_s = new_potential_other_offset_s
                        segment_assembly_plan.append({'type': 'good', 'main_start_s': current_main_time_s, 'duration_s': actual_segment_duration_s, 'other_start_abs_s': other_segment_start_abs_s, 'other_duration_clipped_s': other_segment_duration_clipped_s, 'final_offset_s': current_other_offset_s, 'segment_sync_offset_s': precise_offset_s_fft, 'fidelity': fidelity})
                    else:
                        reason = 'low_fidelity' if fidelity < FIDELITY_THRESHOLD else 'offset_jump'
                        segment_assembly_plan.append({'type': 'bad', 'main_start_s': current_main_time_s, 'duration_s': actual_segment_duration_s, 'reason': reason, 'fidelity': fidelity, 'offset_drift_s': offset_drift_s})
                        total_bad_segments_duration_s += actual_segment_duration_s
                except Exception as e:
                    sys.stderr.write(f"Err correlating for main time {main_segment_start_str}: {e}\n")
                    segment_assembly_plan.append({'type': 'bad', 'main_start_s': current_main_time_s, 'duration_s': actual_segment_duration_s, 'reason': 'correlation_fail'})
                    total_bad_segments_duration_s += actual_segment_duration_s
        total_processed_duration_s += actual_segment_duration_s
        current_main_time_s += actual_segment_duration_s
    
    cleanup_video_obj_temp_audio([main_video_obj, other_video_obj]) # Cleanup audio files from extract_audio_in_part

    if not segment_assembly_plan or (main_total_duration_s > 0 and (total_bad_segments_duration_s / main_total_duration_s * 100.0) > INCOMPATIBILITY_THRESHOLD_PERCENT):
        sys.stderr.write(f"Chimeric generation failed or incompatibility threshold exceeded. Bad segments: {total_bad_segments_duration_s / main_total_duration_s * 100.0:.2f}%\n")
        return None

    # --- FFmpeg Assembly Logic ---
    audio_segment_files = []
    subtitle_segment_files = {} # lang: [paths]
    files_to_cleanup = []
    
    # Sanitize base name for chimeric file
    safe_other_basename = re.sub(r'[^\w\.-]', '_', other_video_obj.fileBaseName)
    chimeric_mkv_path = path.join(out_folder_for_chimeric_file, f"{safe_other_basename}_chimeric_{sync_language}.mkv")

    # Determine primary audio stream order for the sync_language (used for extraction)
    main_primary_audio_stream_order = main_video_obj.audios[sync_language][0]['StreamOrder'] if sync_language in main_video_obj.audios and main_video_obj.audios[sync_language] else None
    other_primary_audio_stream_order = other_video_obj.audios[sync_language][0]['StreamOrder'] if sync_language in other_video_obj.audios and other_video_obj.audios[sync_language] else None
    if not main_primary_audio_stream_order or not other_primary_audio_stream_order:
        sys.stderr.write(f"Primary audio stream for language {sync_language} not found in one or both videos.\n")
        # No specific cleanup here as initial offset files already cleaned, and segment loop files cleaned by its own cleanup_video_obj_temp_audio
        return None 

    all_subtitle_langs = list(set(main_video_obj.subtitles.keys()) | set(other_video_obj.subtitles.keys()))

    for idx, segment_info in enumerate(segment_assembly_plan):
        seg_main_start_s = segment_info['main_start_s']
        seg_duration_s = segment_info['duration_s']
        temp_audio_segment_path = path.join(tools_tmpFolder, f"ch_audio_seg_{idx}.wav")
        files_to_cleanup.append(temp_audio_segment_path)

        source_obj_audio = other_video_obj if segment_info['type'] == 'good' else main_video_obj
        audio_ss = segment_info['other_start_abs_s'] if segment_info['type'] == 'good' else seg_main_start_s
        audio_t = segment_info['other_duration_clipped_s'] if segment_info['type'] == 'good' else seg_duration_s
        audio_stream_order_to_extract = other_primary_audio_stream_order if segment_info['type'] == 'good' else main_primary_audio_stream_order
        
        # Ensure audio_ss is not negative if it comes from other_start_abs_s
        audio_ss = max(0.0, audio_ss)

        if not extract_audio_segment(source_obj_audio.filePath, audio_stream_order_to_extract, format_time_ffmpeg(audio_ss), format_time_ffmpeg(audio_t), temp_audio_segment_path):
            sys.stderr.write(f"Failed to extract audio segment {idx}, aborting chimeric assembly.\n")
            cleanup_temp_files(files_to_cleanup)
            return None
        audio_segment_files.append(temp_audio_segment_path)

        for sub_lang in all_subtitle_langs:
            temp_sub_segment_path = path.join(tools_tmpFolder, f"ch_sub_seg_{sub_lang}_{idx}.srt")
            files_to_cleanup.append(temp_sub_segment_path)
            subtitle_segment_files.setdefault(sub_lang, []).append(temp_sub_segment_path)
            
            extracted_successfully = False
            source_obj_subs = other_video_obj if segment_info['type'] == 'good' else main_video_obj
            subs_ss = segment_info['other_start_abs_s'] if segment_info['type'] == 'good' else seg_main_start_s
            subs_t = segment_info['other_duration_clipped_s'] if segment_info['type'] == 'good' else seg_duration_s
            subs_ss = max(0.0, subs_ss) # Ensure positive start time

            if sub_lang in source_obj_subs.subtitles and source_obj_subs.subtitles[sub_lang]:
                # Prioritize 'keep=True' subs if any, otherwise take first available.
                chosen_sub_track_info = None
                kept_subs = [st for st in source_obj_subs.subtitles[sub_lang] if st.get('keep', True)]
                if kept_subs:
                    chosen_sub_track_info = kept_subs[0] # Take the first 'kept' sub
                elif source_obj_subs.subtitles[sub_lang]: # Fallback to first sub if no 'kept' ones
                    chosen_sub_track_info = source_obj_subs.subtitles[sub_lang][0]

                if chosen_sub_track_info:
                    extracted_successfully = extract_specific_subtitle_stream_segment(source_obj_subs.filePath, chosen_sub_track_info['StreamOrder'], subs_ss, subs_t, temp_sub_segment_path)
            
            if not extracted_successfully:
                generate_blank_sub_segment(subs_t, temp_sub_segment_path) # Use clipped/actual duration for blank

    # Create Concat Files
    audio_concat_list_path = path.join(tools_tmpFolder, "ch_audio_concat.txt")
    with open(audio_concat_list_path, 'w', encoding='utf-8') as f:
        for p in audio_segment_files: f.write(f"file '{path.basename(p)}'\n") # relative path for ffmpeg concat
    files_to_cleanup.append(audio_concat_list_path)

    subtitle_concat_list_paths = {}
    for lang, file_list in subtitle_segment_files.items():
        sub_concat_path = path.join(tools_tmpFolder, f"ch_sub_{lang}_concat.txt")
        with open(sub_concat_path, 'w', encoding='utf-8') as f:
            for p in file_list: f.write(f"file '{path.basename(p)}'\n") # relative path
        subtitle_concat_list_paths[lang] = sub_concat_path
        files_to_cleanup.append(sub_concat_path)

    # FFmpeg Command Assembly
    ffmpeg_cmd = [tools.software['ffmpeg'], '-y', '-nostdin']
    ffmpeg_cmd.extend(['-i', other_video_obj.filePath]) # Input 0: Video from other_video_obj
    ffmpeg_cmd.extend(['-f', 'concat', '-safe', '0', '-i', audio_concat_list_path]) # Input 1: Concatenated Audio
    
    input_idx_counter = 2 # Starts at 2 because 0 is video, 1 is audio
    subtitle_maps_metadata = []
    for lang in all_subtitle_langs: # Ensure consistent order for mapping
        if lang in subtitle_concat_list_paths:
            ffmpeg_cmd.extend(['-f', 'concat', '-safe', '0', '-i', subtitle_concat_list_paths[lang]])
            subtitle_maps_metadata.append(({'map_idx': input_idx_counter, 'lang': lang}))
            input_idx_counter += 1
    
    ffmpeg_cmd.extend(['-map', '0:v:0', '-map', '1:a:0'])
    for i, sub_map_info in enumerate(subtitle_maps_metadata):
        ffmpeg_cmd.extend(['-map', f"{sub_map_info['map_idx']}:s:0"])
        ffmpeg_cmd.extend([f"-metadata:s:s:{i}", f"language={sub_map_info['lang']}"]) # s:{i} is output subtitle stream index

    ffmpeg_cmd.extend(['-c:v', 'copy', '-c:a', 'flac', '-c:s', 'ass']) # Encode audio to FLAC, subs to ASS
    ffmpeg_cmd.extend(['-shortest', chimeric_mkv_path]) # Use -shortest to ensure output duration matches video

    final_return_path = None
    try:
        sys.stdout.write(f"Executing FFmpeg command for chimeric video: {' '.join(ffmpeg_cmd)}\n")
        _, stderr_ffmpeg, exit_code_ffmpeg = tools.launch_cmdExt(ffmpeg_cmd, cwd=tools_tmpFolder) # Run in tmpFolder for relative paths
        if exit_code_ffmpeg == 0:
            sys.stdout.write(f"Chimeric MKV generated: {chimeric_mkv_path}\n")
            final_return_path = chimeric_mkv_path
        else:
            sys.stderr.write(f"Error generating chimeric MKV: {stderr_ffmpeg.decode('utf-8', errors='ignore')}\n")
    except Exception as e:
        sys.stderr.write(f"Exception during FFmpeg execution for chimeric video: {e}\n")
    finally:
        cleanup_temp_files(files_to_cleanup)
        
    return final_return_path

def extract_audio_segment(video_obj_path, audio_track_stream_order, start_time_str, duration_str, output_wav_path):
    """Extracts a specific audio track segment to output_wav_path as WAV PCM S16LE."""
    try:
        ffmpeg_cmd = [
            tools.software['ffmpeg'], '-y',
            '-i', video_obj_path,
            '-ss', start_time_str,
            '-t', duration_str,
            '-map', f"0:{audio_track_stream_order}",
            '-vn', '-acodec', 'pcm_s16le', '-ar', '48000', '-ac', '2',
            output_wav_path
        ]
        stdout, stderr_str, exit_code = tools.launch_cmdExt(ffmpeg_cmd)
        if exit_code != 0:
            sys.stderr.write(f"Error extracting audio segment {output_wav_path}: {stderr_str.decode('utf-8', errors='ignore')}\n")
            return False
        return True
    except Exception as e:
        sys.stderr.write(f"Exception extracting audio segment {output_wav_path}: {e}\n")
        return False

def generate_blank_sub_segment(duration_s, output_srt_path):
    """Creates a blank SRT file of duration_s at output_srt_path."""
    try:
        formatted_duration = video.seconds_to_srt_timeformat(duration_s) # Assuming video.seconds_to_srt_timeformat
        blank_srt_content = f"1\n00:00:00,000 --> {formatted_duration}\n\n"
        with open(output_srt_path, 'w', encoding='utf-8') as f:
            f.write(blank_srt_content)
        return True
    except Exception as e:
        sys.stderr.write(f"Error generating blank subtitle {output_srt_path}: {e}\n")
        return False

def extract_specific_subtitle_stream_segment(video_file_path, stream_order_id, start_s_float, duration_s_float, output_srt_path):
    """Extracts a specific subtitle stream segment as SRT."""
    try:
        # Ensure start_s and duration_s are formatted correctly for ffmpeg if they are floats
        start_s_str = format_time_ffmpeg(start_s_float)
        duration_s_str = format_time_ffmpeg(duration_s_float)

        ffmpeg_cmd = [
            tools.software['ffmpeg'], '-y',
            '-i', video_file_path,
            '-ss', start_s_str,
            '-t', duration_s_str,
            '-map', f"0:{stream_order_id}",
            '-c:s', 'srt',
            output_srt_path
        ]
        stdout, stderr_str, exit_code = tools.launch_cmdExt(ffmpeg_cmd)
        if exit_code != 0:
            # It's common for subtitle extraction to fail if no subs in range, not always a fatal error for the process.
            # Consider logging this as info/warning rather than error if that's acceptable.
            # For now, treating as a failure to extract that specific segment.
            # sys.stderr.write(f"Warning/Error extracting subtitle segment {output_srt_path}: {stderr_str.decode('utf-8', errors='ignore')}\n")
            return False
        return True
    except Exception as e:
        sys.stderr.write(f"Exception extracting subtitle segment {output_srt_path}: {e}\n")
        return False

def generate_chimeric_video(main_video_obj, other_video_obj, initial_offset_ms, sync_language, audio_param_for_extraction, tools_tmpFolder, out_folder_for_chimeric_file):
    FIDELITY_THRESHOLD = 0.80
    SEGMENT_DURATION_S = 60.0
    INCOMPATIBILITY_THRESHOLD_PERCENT = 50.0
    OFFSET_CONSISTENCY_WINDOW_S = 0.5

    # Ensure `tools_tmpFolder` is available (passed as argument)

    segment_assembly_plan = []
    current_main_time_s = 0.0
    if not isinstance(initial_offset_ms, Decimal):
        initial_offset_ms = Decimal(str(initial_offset_ms))
    current_other_offset_s = float(initial_offset_ms / Decimal(1000))

    try:
        main_total_duration_s = float(main_video_obj.video['Duration'])
        other_total_duration_s = float(other_video_obj.video['Duration'])
    except (KeyError, ValueError, TypeError) as e:
        sys.stderr.write(f"Error getting video durations: {e}\n")
        cleanup_video_obj_temp_audio([main_video_obj, other_video_obj]) # Cleanup before early exit
        return None

    total_bad_segments_duration_s = 0.0
    total_processed_duration_s = 0.0

    # --- Segment processing loop (largely as before) ---
    while current_main_time_s < main_total_duration_s:
        actual_segment_duration_s = min(SEGMENT_DURATION_S, main_total_duration_s - current_main_time_s)
        if actual_segment_duration_s < 1.0: break
        main_segment_start_str = format_time_ffmpeg(current_main_time_s)
        main_segment_duration_str = format_time_ffmpeg(actual_segment_duration_s)
        main_cut_time = [[main_segment_start_str, main_segment_duration_str]]
        try:
            main_video_obj.extract_audio_in_part(sync_language, audio_param_for_extraction, cutTime=main_cut_time, asDefault=True)
            if not main_video_obj.tmpFiles.get('audio') or not main_video_obj.tmpFiles['audio'][0]: raise Exception("Main audio seg path not found.")
            main_audio_segment_path = main_video_obj.tmpFiles['audio'][0][0]
        except Exception as e:
            sys.stderr.write(f"Err extracting main audio {main_segment_start_str}: {e}\n")
            segment_assembly_plan.append({'type': 'bad', 'main_start_s': current_main_time_s, 'duration_s': actual_segment_duration_s, 'reason': 'main_extract_fail'})
            total_bad_segments_duration_s += actual_segment_duration_s
            total_processed_duration_s += actual_segment_duration_s; current_main_time_s += actual_segment_duration_s; continue
        other_segment_start_abs_s = current_main_time_s - current_other_offset_s
        if (other_segment_start_abs_s + actual_segment_duration_s < 0) or (other_segment_start_abs_s >= other_total_duration_s):
            segment_assembly_plan.append({'type': 'bad', 'main_start_s': current_main_time_s, 'duration_s': actual_segment_duration_s, 'reason': 'other_out_of_bounds'})
            total_bad_segments_duration_s += actual_segment_duration_s
        else:
            other_segment_start_clipped_s = max(0.0, other_segment_start_abs_s)
            other_segment_duration_clipped_s = min(actual_segment_duration_s, other_total_duration_s - other_segment_start_clipped_s)
            if other_segment_duration_clipped_s < 1.0:
                segment_assembly_plan.append({'type': 'bad', 'main_start_s': current_main_time_s, 'duration_s': actual_segment_duration_s, 'reason': 'other_clip_too_short'})
                total_bad_segments_duration_s += actual_segment_duration_s
            else:
                other_segment_start_str = format_time_ffmpeg(other_segment_start_clipped_s); other_segment_duration_str = format_time_ffmpeg(other_segment_duration_clipped_s)
                other_cut_time = [[other_segment_start_str, other_segment_duration_str]]
                try:
                    other_video_obj.extract_audio_in_part(sync_language, audio_param_for_extraction, cutTime=other_cut_time, asDefault=True)
                    if not other_video_obj.tmpFiles.get('audio') or not other_video_obj.tmpFiles['audio'][0]: raise Exception("Other audio seg path not found.")
                    other_audio_segment_path = other_video_obj.tmpFiles['audio'][0][0]
                    fidelity, _, _ = correlate(main_audio_segment_path, other_audio_segment_path, other_segment_duration_clipped_s)
                    _shifted_file, precise_offset_s_fft_raw = second_correlation(main_audio_segment_path, other_audio_segment_path)
                    precise_offset_s_fft = -precise_offset_s_fft_raw if _shifted_file == main_audio_segment_path else precise_offset_s_fft_raw
                    new_potential_other_offset_s = current_other_offset_s - precise_offset_s_fft
                    offset_drift_s = abs(new_potential_other_offset_s - current_other_offset_s)
                    if fidelity >= FIDELITY_THRESHOLD and offset_drift_s <= OFFSET_CONSISTENCY_WINDOW_S:
                        current_other_offset_s = new_potential_other_offset_s
                        segment_assembly_plan.append({'type': 'good', 'main_start_s': current_main_time_s, 'duration_s': actual_segment_duration_s, 'other_start_abs_s': other_segment_start_abs_s, 'other_duration_clipped_s': other_segment_duration_clipped_s, 'final_offset_s': current_other_offset_s, 'segment_sync_offset_s': precise_offset_s_fft, 'fidelity': fidelity})
                    else:
                        reason = 'low_fidelity' if fidelity < FIDELITY_THRESHOLD else 'offset_jump'
                        segment_assembly_plan.append({'type': 'bad', 'main_start_s': current_main_time_s, 'duration_s': actual_segment_duration_s, 'reason': reason, 'fidelity': fidelity, 'offset_drift_s': offset_drift_s})
                        total_bad_segments_duration_s += actual_segment_duration_s
                except Exception as e:
                    sys.stderr.write(f"Err correlating for main time {main_segment_start_str}: {e}\n")
                    segment_assembly_plan.append({'type': 'bad', 'main_start_s': current_main_time_s, 'duration_s': actual_segment_duration_s, 'reason': 'correlation_fail'})
                    total_bad_segments_duration_s += actual_segment_duration_s
        total_processed_duration_s += actual_segment_duration_s
        current_main_time_s += actual_segment_duration_s
    
    cleanup_video_obj_temp_audio([main_video_obj, other_video_obj]) # Cleanup audio files from extract_audio_in_part

    if not segment_assembly_plan or (main_total_duration_s > 0 and (total_bad_segments_duration_s / main_total_duration_s * 100.0) > INCOMPATIBILITY_THRESHOLD_PERCENT):
        sys.stderr.write(f"Chimeric generation failed or incompatibility threshold exceeded. Bad segments: {total_bad_segments_duration_s / main_total_duration_s * 100.0:.2f}%\n")
        return None

    # --- FFmpeg Assembly Logic ---
    audio_segment_files = []
    subtitle_segment_files = {} # lang: [paths]
    files_to_cleanup = []
    
    # Sanitize base name for chimeric file
    safe_other_basename = re.sub(r'[^\w\.-]', '_', other_video_obj.fileBaseName)
    chimeric_mkv_path = path.join(out_folder_for_chimeric_file, f"{safe_other_basename}_chimeric_{sync_language}.mkv")

    # Determine primary audio stream order for the sync_language (used for extraction)
    main_primary_audio_stream_order = main_video_obj.audios[sync_language][0]['StreamOrder'] if sync_language in main_video_obj.audios and main_video_obj.audios[sync_language] else None
    other_primary_audio_stream_order = other_video_obj.audios[sync_language][0]['StreamOrder'] if sync_language in other_video_obj.audios and other_video_obj.audios[sync_language] else None
    if not main_primary_audio_stream_order or not other_primary_audio_stream_order:
        sys.stderr.write(f"Primary audio stream for language {sync_language} not found in one or both videos.\n")
        return None # Cannot proceed without audio streams to sync with

    all_subtitle_langs = list(set(main_video_obj.subtitles.keys()) | set(other_video_obj.subtitles.keys()))

    for idx, segment_info in enumerate(segment_assembly_plan):
        seg_main_start_s = segment_info['main_start_s']
        seg_duration_s = segment_info['duration_s']
        temp_audio_segment_path = path.join(tools_tmpFolder, f"ch_audio_seg_{idx}.wav")
        files_to_cleanup.append(temp_audio_segment_path)

        source_obj_audio = other_video_obj if segment_info['type'] == 'good' else main_video_obj
        audio_ss = segment_info['other_start_abs_s'] if segment_info['type'] == 'good' else seg_main_start_s
        audio_t = segment_info['other_duration_clipped_s'] if segment_info['type'] == 'good' else seg_duration_s
        audio_stream_order_to_extract = other_primary_audio_stream_order if segment_info['type'] == 'good' else main_primary_audio_stream_order
        
        # Ensure audio_ss is not negative if it comes from other_start_abs_s
        audio_ss = max(0.0, audio_ss)

        if not extract_audio_segment(source_obj_audio.filePath, audio_stream_order_to_extract, format_time_ffmpeg(audio_ss), format_time_ffmpeg(audio_t), temp_audio_segment_path):
            sys.stderr.write(f"Failed to extract audio segment {idx}, aborting chimeric assembly.\n")
            cleanup_temp_files(files_to_cleanup)
            return None
        audio_segment_files.append(temp_audio_segment_path)

        for sub_lang in all_subtitle_langs:
            temp_sub_segment_path = path.join(tools_tmpFolder, f"ch_sub_seg_{sub_lang}_{idx}.srt")
            files_to_cleanup.append(temp_sub_segment_path)
            subtitle_segment_files.setdefault(sub_lang, []).append(temp_sub_segment_path)
            
            extracted_successfully = False
            source_obj_subs = other_video_obj if segment_info['type'] == 'good' else main_video_obj
            subs_ss = segment_info['other_start_abs_s'] if segment_info['type'] == 'good' else seg_main_start_s
            subs_t = segment_info['other_duration_clipped_s'] if segment_info['type'] == 'good' else seg_duration_s
            subs_ss = max(0.0, subs_ss) # Ensure positive start time

            if sub_lang in source_obj_subs.subtitles and source_obj_subs.subtitles[sub_lang]:
                # Prioritize 'keep=True' subs if any, otherwise take first available.
                chosen_sub_track_info = None
                kept_subs = [st for st in source_obj_subs.subtitles[sub_lang] if st.get('keep', True)]
                if kept_subs:
                    chosen_sub_track_info = kept_subs[0] # Take the first 'kept' sub
                elif source_obj_subs.subtitles[sub_lang]: # Fallback to first sub if no 'kept' ones
                    chosen_sub_track_info = source_obj_subs.subtitles[sub_lang][0]

                if chosen_sub_track_info:
                    extracted_successfully = extract_specific_subtitle_stream_segment(source_obj_subs.filePath, chosen_sub_track_info['StreamOrder'], subs_ss, subs_t, temp_sub_segment_path)
            
            if not extracted_successfully:
                generate_blank_sub_segment(subs_t, temp_sub_segment_path) # Use clipped/actual duration for blank

    # Create Concat Files
    audio_concat_list_path = path.join(tools_tmpFolder, "ch_audio_concat.txt")
    with open(audio_concat_list_path, 'w', encoding='utf-8') as f:
        for p in audio_segment_files: f.write(f"file '{path.basename(p)}'\n") # relative path for ffmpeg concat
    files_to_cleanup.append(audio_concat_list_path)

    subtitle_concat_list_paths = {}
    for lang, file_list in subtitle_segment_files.items():
        sub_concat_path = path.join(tools_tmpFolder, f"ch_sub_{lang}_concat.txt")
        with open(sub_concat_path, 'w', encoding='utf-8') as f:
            for p in file_list: f.write(f"file '{path.basename(p)}'\n") # relative path
        subtitle_concat_list_paths[lang] = sub_concat_path
        files_to_cleanup.append(sub_concat_path)

    # FFmpeg Command Assembly
    ffmpeg_cmd = [tools.software['ffmpeg'], '-y', '-nostdin']
    ffmpeg_cmd.extend(['-i', other_video_obj.filePath]) # Input 0: Video from other_video_obj
    ffmpeg_cmd.extend(['-f', 'concat', '-safe', '0', '-i', audio_concat_list_path]) # Input 1: Concatenated Audio
    
    input_idx_counter = 2 # Starts at 2 because 0 is video, 1 is audio
    subtitle_maps_metadata = []
    for lang in all_subtitle_langs: # Ensure consistent order for mapping
        if lang in subtitle_concat_list_paths:
            ffmpeg_cmd.extend(['-f', 'concat', '-safe', '0', '-i', subtitle_concat_list_paths[lang]])
            subtitle_maps_metadata.append(({'map_idx': input_idx_counter, 'lang': lang}))
            input_idx_counter += 1
    
    ffmpeg_cmd.extend(['-map', '0:v:0', '-map', '1:a:0'])
    for i, sub_map_info in enumerate(subtitle_maps_metadata):
        ffmpeg_cmd.extend(['-map', f"{sub_map_info['map_idx']}:s:0"])
        ffmpeg_cmd.extend([f"-metadata:s:s:{i}", f"language={sub_map_info['lang']}"]) # s:{i} is output subtitle stream index

    ffmpeg_cmd.extend(['-c:v', 'copy', '-c:a', 'flac', '-c:s', 'ass']) # Encode audio to FLAC, subs to ASS
    ffmpeg_cmd.extend(['-shortest', chimeric_mkv_path]) # Use -shortest to ensure output duration matches video

    final_return_path = None
    try:
        sys.stdout.write(f"Executing FFmpeg command for chimeric video: {' '.join(ffmpeg_cmd)}\n")
        _, stderr_ffmpeg, exit_code_ffmpeg = tools.launch_cmdExt(ffmpeg_cmd, cwd=tools_tmpFolder) # Run in tmpFolder for relative paths
        if exit_code_ffmpeg == 0:
            sys.stdout.write(f"Chimeric MKV generated: {chimeric_mkv_path}\n")
            final_return_path = chimeric_mkv_path
        else:
            sys.stderr.write(f"Error generating chimeric MKV: {stderr_ffmpeg.decode('utf-8', errors='ignore')}\n")
    except Exception as e:
        sys.stderr.write(f"Exception during FFmpeg execution for chimeric video: {e}\n")
    finally:
        cleanup_temp_files(files_to_cleanup)
        
    return final_return_path