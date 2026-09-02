'''
Code retire de mergeVideo.py: sans appelant dans tout le depot.

Conserve ici plutot que supprime parce qu'il porte une intention -- decouper un
fichier incompatible pour retrouver un point de synchronisation -- que la
campagne a reprise autrement (merge_video_chimeric). Rien n'importe ce module;
il est la pour etre lu, pas pour etre execute.

Verifie avant extraction: aucune reference a `find_a_cut_for_not_compatible` ni
a la classe `get_cut_time` hors de leur propre definition. Attention a
l'homonymie: le module `get_cut_time.py` est un fichier different, lui aussi
sans importateur.
'''

import sys
import traceback
from threading import Thread

import tools
import video


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
