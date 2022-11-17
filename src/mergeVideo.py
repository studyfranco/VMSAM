'''
Created on 24 Apr 2022

@author: francois

This software need libchromaprint-tools,ffmpeg,mediainfo
'''

import argparse
from os import path,remove,chdir
from time import strftime,gmtime

numberCut = 10

def decript_merge_rules(stringRules):
    #video=x266*6>x265*2>x264
    #audio=DTS>E-AAC*300>AAC*10>MP3,DTS=Flac,Flac>E-AAC
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
    
def get_good_parameters_to_get_fidelity(baseVideoObj,language,audioParam,maxTime):
    if maxTime < 10:
        timeTake = strftime('%H:%M:%S',gmtime(maxTime))
    else:
        timeTake = "00:00:10"
    nameFilesAudioCutBaseVideo = baseVideoObj.extract_audio_in_part(language,audioParam,cutTime=[["00:00:00",timeTake]])
    if (not test_calcul_can_be(nameFilesAudioCutBaseVideo[0][0])):
        remove(nameFilesAudioCutBaseVideo[0][0])
        audioParam["Format"] = "AAC"
        audioParam['Channels'] = "2"
        if int(audioParam["BitRate"]) > 256000:
            audioParam["BitRate"] = "256000"
        nameFilesAudioCutBaseVideo = baseVideoObj.extract_audio_in_part(language,audioParam,cutTime=[["00:00:00",timeTake]])
        if (not test_calcul_can_be(nameFilesAudioCutBaseVideo[0][0])):
            remove(nameFilesAudioCutBaseVideo[0][0])
            audioParam["Format"] = "MP3"
        else:
            remove(nameFilesAudioCutBaseVideo[0][0])
    else:
        remove(nameFilesAudioCutBaseVideo[0][0])
    
def get_delay_fidelity_between_files(fileBase,fileToGetDelay, lengthFiles):
    return correlate(fileBase, fileToGetDelay, lengthFiles)

def get_delay_fidelity(nameFilesAudioCutBaseVideo,videoObj,language,audioParam,orginialCutTime,lenghtTime):
    nameFilesAudioCutVideoOther = videoObj.extract_audio_in_part(language,audioParam,cutTime=orginialCutTime)
    delayFidelityValuesAgainstBaseForFiles = []
    for i in range(0,numberCut):
        delayFidelityValuesAgainstBaseForFile = []
        delayFidelityValuesAgainstBaseForFiles.append(delayFidelityValuesAgainstBaseForFile)
        for nameFileAudioCutVideoOther in nameFilesAudioCutVideoOther[i]:
            delayFidelityValuesAgainstBaseForFile.append(get_delay_fidelity_between_files(nameFilesAudioCutBaseVideo[i][0],nameFileAudioCutVideoOther,lenghtTime))
            remove(nameFileAudioCutVideoOther)
    return delayFidelityValuesAgainstBaseForFiles

def test_if_constant_good_delay(nameFilesAudioCutBaseVideo,videoObj,language,audioParam,orginialCutTime,beginInSec,lenghtTime,lenghtTimePrepare):
    delayFidelityValuesAgainstBaseForFiles = get_delay_fidelity(nameFilesAudioCutBaseVideo,videoObj,language,audioParam,orginialCutTime,lenghtTime)
    dictDelay = {}
    
    for delayFidelityValuesAgainstBaseForFile in delayFidelityValuesAgainstBaseForFiles:
        for delayFidelityValuesAgainstBase in delayFidelityValuesAgainstBaseForFile:
            if delayFidelityValuesAgainstBase[2] in dictDelay:
                dictDelay[delayFidelityValuesAgainstBase[2]] += 1
            else:
                dictDelay[delayFidelityValuesAgainstBase[2]] = 1
    
    if len(dictDelay) != 1:
        raise Exception("Multiple delay found in the file {} against the base".format(videoObj.filePath))
    else:
        delayUse = list(dictDelay.keys())[0]
    
    delayUseNegative = delayUse < 0
    delayUseSec = int(abs(delayUse)/1000)
    delayUseMillisecond = abs(delayUse)%1000
    cutTime = []
    if delayUseNegative:
        if delayUseSec > beginInSec or (delayUseSec == beginInSec and delayUseMillisecond > 0):
            raise Exception("Need to be update for negative delay and negative Time")
        else:
            if delayUseMillisecond > 0:
                beginInSec -= 1
                delayUseMillisecond = 1000-delayUseMillisecond
                if delayUseMillisecond < 10:
                    zeroBefore = "0"*2
                elif delayUseMillisecond < 100:
                    zeroBefore = "0"*1
                else:
                    zeroBefore = ""
            beginInSec -= delayUseSec
            for i in range(0,numberCut):
                cutTime.append([strftime('%H:%M:%S',gmtime(beginInSec))+"."+zeroBefore+str(delayUseMillisecond),lenghtTimePrepare])
                beginInSec += lenghtTime
    else:
        beginInSec += delayUseSec
        if delayUseMillisecond < 10:
            zeroBefore = "0"*2
        elif delayUseMillisecond < 100:
            zeroBefore = "0"*1
        else:
            zeroBefore = ""
        for i in range(0,numberCut):
            cutTime.append([strftime('%H:%M:%S',gmtime(beginInSec))+"."+zeroBefore+str(delayUseMillisecond),lenghtTimePrepare])
            beginInSec += lenghtTime
    
    delayFidelityValuesAgainstBaseForFiles = get_delay_fidelity(nameFilesAudioCutBaseVideo,videoObj,language,audioParam,cutTime,lenghtTime)
    dictDelay = {}
    
    for delayFidelityValuesAgainstBaseForFile in delayFidelityValuesAgainstBaseForFiles:
        for delayFidelityValuesAgainstBase in delayFidelityValuesAgainstBaseForFile:
            if delayFidelityValuesAgainstBase[2] in dictDelay:
                dictDelay[delayFidelityValuesAgainstBase[2]] += 1
            else:
                dictDelay[delayFidelityValuesAgainstBase[2]] = 1
                
    if len(dictDelay) == 1 and 0 in dictDelay:
        return delayUse
    else:
        raise Exception("Not able to find delay for {}".format(videoObj.filePath))
    
        
def mergeVideo(files,outFolder,inFolder=None):
    videosObj = []    
    if inFolder == None:
        for file in files:
            videosObj.append(video(path.dirname(file),path.basename(file)))
    else:
        for file in files:
            videosObj.append(video(inFolder,file))
            
    mergeRules = tools.config_loader(args.config,"mergerules")
    audioRules = decript_merge_rules(mergeRules['audio'])
    videoRules = decript_merge_rules(mergeRules['video'])
    
    for videoObj in videosObj:
        videoObj.get_mediadata()

    commonLanguages = get_common_audios_language(videosObj)
    if len(commonLanguages) == 0:
        raise Exception("No common language between "+str([videoObj.filePath for videoObj in videosObj]))
    allVideosObj = videosObj.copy()
    baseVideoObj = videosObj.pop(get_ID_best_quality_video(videosObj,videoRules))
    for language in commonLanguages:
        worseAudioQualityWillUse = get_worse_quality_audio_param(allVideosObj,language,audioRules)
        minAudioTimeInSec = get_shortest_audio_durations(videosObj,language)
        get_good_parameters_to_get_fidelity(baseVideoObj,language,worseAudioQualityWillUse,minAudioTimeInSec)
        if minAudioTimeInSec > 600:
            beginInSec = 180
            lenghtTime = int((minAudioTimeInSec-180)/numberCut)
        else:
            beginInSec = 5
            lenghtTime = int((minAudioTimeInSec-5)/numberCut)
        cutTime = []
        lenghtTimePrepare = strftime('%H:%M:%S',gmtime(lenghtTime))
        beginInSecOriginal = beginInSec
        for i in range(0,numberCut):
            cutTime.append([strftime('%H:%M:%S',gmtime(beginInSec)),lenghtTimePrepare])
            beginInSec += lenghtTime
        nameFilesAudioCutBaseVideo = baseVideoObj.extract_audio_in_part(language,worseAudioQualityWillUse,cutTime=cutTime)
        delays = []
        for videoObj in videosObj:
            delays.append(test_if_constant_good_delay(nameFilesAudioCutBaseVideo,videoObj,language,worseAudioQualityWillUse,cutTime,beginInSecOriginal,lenghtTime,lenghtTimePrepare))

        for nameFilesAudioCutsBaseVideo in nameFilesAudioCutBaseVideo:
            for nameFileAudioCutBaseVideo in nameFilesAudioCutsBaseVideo:
                remove(nameFileAudioCutBaseVideo)
        
    
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
    
    software = tools.config_loader(args.config, "software")
    chdir(args.pwd)
    import tools
    from video import video, get_ID_best_quality_video, get_common_audios_language, get_worse_quality_audio_param, get_shortest_audio_durations
    from audioCorrelation import correlate, test_calcul_can_be
    tools.tmpFolder = args.tmp
    
    mergeVideo(args.file.split(","), args.out, args.folder)