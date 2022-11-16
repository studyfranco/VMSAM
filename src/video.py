'''
Created on 23 Apr 2022

@author: francois
'''

from os import path
import tools
import json

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
    
    def get_mediadata(self):
        stdout, stderror, exitCode = tools.launch_cmdExt([tools.software["mediainfo"], "--Output=JSON", self.filePath])
        if exitCode != 0:
            raise Exception("Error with {} during the mediadata: {}".format(self.filePath,stderror.decode("UTF-8")))
        self.mediadata = json.loads(stdout.decode("UTF-8"))
        self.audios = {}
        self.subtitles = {}
        audioWithoutLanguage = []
        for data in self.mediadata['media']['track']:
            if data['@type'] == 'Video':
                if self.video != None:
                    raise Exception(self.filePath+" multiple video in same file")
                self.video = data
            elif 'Language' in data:
                if data['@type'] == 'Audio':
                    if data['Language'] in self.audios:
                        self.audios[data['Language']].append(data)
                    else:
                        self.audios[data['Language']] = [data]
                elif data['@type'] == 'Text':
                    if data['Language'] in self.subtitles:
                        self.subtitles[data['Language']].append(data)
                    else:
                        self.subtitles[data['Language']] = [data]
            elif data['@type'] == 'Audio':
                audioWithoutLanguage.append(data)
        if len(self.audios) == 0 and len(audioWithoutLanguage) == 1:
            self.audios["ja"] = [audioWithoutLanguage[0]]
    
    def extract_audio_in_part(self,language,exportParam,cutTime=None):
        nameFilesExtract = []
        baseCommand = [tools.software["ffmpeg"], "-nostdin", "-i", self.filePath, "-vn", "-acodec", exportParam["Format"].lower().replace('-',''), "-ab", exportParam["BitRate"], "-ar", exportParam['SamplingRate']]
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
                stdout, stderror, exitCode = tools.launch_cmdExt(cmd)
                #print("{}\n{}\n{}\n".format(stdout, stderror, exitCode))
                #print(" ".join(cmd))
        else:
            cutNumber = 0
            for cut in cutTime:
                nameFilesExtractCut = []
                nameFilesExtract.append(nameFilesExtractCut)
                for audio in self.audios[language]:
                    nameOutFile = path.join(tools.tmpFolder,self.fileBaseName+"."+str(audio["ID"])+"."+str(cutNumber)+"."+exportParam['Format'].lower().replace('-',''))
                    nameFilesExtractCut.append(nameOutFile)
                    cmd = baseCommand.copy()
                    cmd.extend(["-map", "0:"+str(int(audio["ID"])-1), "-ss", cut[0], "-t", cut[1] , nameOutFile])
                    stdout, stderror, exitCode = tools.launch_cmdExt(cmd)
                cutNumber += 1
                
        return nameFilesExtract

# Une amelioration serait de faire des RULES paramétrable. A y réfléchir.
def get_ID_best_quality_video(videosObj,rules):
    bestVideo = 0
    for i in range(1,len(videosObj)):
        if test_if_the_best_by_rules_video_entry(videosObj[bestVideo].video,videosObj[i].video,rules):
            bestVideo = i
    return bestVideo

def get_common_audios_language(videosObj):
    commonLanguages = set(videosObj[0].audios.keys())
    for videoObj in videosObj:
        commonLanguages.union(videoObj.audios.keys())
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