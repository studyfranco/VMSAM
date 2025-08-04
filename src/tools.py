'''
Created on 23 Apr 2022

@author: francois
'''
import os
import shutil
import sys
from subprocess import Popen, PIPE, TimeoutExpired
import psutil
import time
from configparser import ConfigParser

def config_loader(file, section):
    parser = ConfigParser()
    parser.read(file)

    # get section
    infos = {}
    if parser.has_section(section):
        params = parser.items(section)
        for param in params:
            infos[param[0]] = param[1]
    else:
        raise Exception("Section "+section+" not found in the "+file+" file")
 
    return infos

''' Files functions '''
def file_exists(f):
    try:
        with open(f):
            return True
    except IOError:
        return False
    
def file_remove(path,file):
    os.remove(os.path.join(path,file))

def make_dirs(d):
    try:
        os.makedirs(d,exist_ok=True)
        return os.path.isdir(d)
    except:
        return False
    
def move_dir(Dir,Folder,raise_exception=True):
    try:
        shutil.move(Dir,Folder)
        return True,None
    except Exception as e:
        if raise_exception:
            raise e
        else:
            return False,e
    
def remove_dir(dir_path,printError=True):
    try:
        shutil.rmtree(dir_path)
    except OSError as e:
        if printError:
            sys.stderr.write("Error: %s : %s\n" % (dir_path, e.strerror))

''' Popen functions '''
def launch_cmdExt(cmd):
    cmdDownload = Popen(cmd, stdout=PIPE, stderr=PIPE)
    stdout, stderror = cmdDownload.communicate()
    exitCode = cmdDownload.returncode
    if exitCode != 0:
        raise Exception("This cmd is in error: "+" ".join(cmd)+"\n"+str(stderror.decode("utf-8"))+"\n"+str(stdout.decode("utf-8"))+"\nReturn code: "+str(exitCode)+"\n")
    return stdout, stderror, exitCode

def launch_cmdExt_no_test(cmd):
    cmdDownload = Popen(cmd, stdout=PIPE, stderr=PIPE)
    stdout, stderror = cmdDownload.communicate()
    exitCode = cmdDownload.returncode
    return stdout, stderror, exitCode

def launch_cmdExt_with_tester(cmd,max_restart=1,timeout=120):
    cmdDownload = Popen(cmd, stdout=PIPE, stderr=PIPE)
    try:
        ps_proc = psutil.Process(cmdDownload.pid)
        start_time = time.time()
        
        while cmdDownload.poll() == None:
            time.sleep(10)
            if ps_proc.status() == psutil.STATUS_ZOMBIE or ps_proc.cpu_percent(interval=1.0) < 0.1:
                if ps_proc.cpu_percent(interval=2.0) < 0.1 and cmdDownload.poll() == None:
                    try:
                        cmdDownload.kill()
                    except Exception:
                        pass
                    try:
                        cmdDownload.communicate(timeout=5)
                    except TimeoutExpired:
                        try:
                            cmdDownload.kill()
                        except:
                            pass
                    
                    max_restart -= 1
                    if max_restart < 0:
                        raise Exception("The process is zombie and cannot be restarted: "+" ".join(cmd)+"\n")
                    else:
                        cmdDownload = Popen(cmd, stdout=PIPE, stderr=PIPE)
                        ps_proc = psutil.Process(cmdDownload.pid)
                        start_time = time.time()
            elif time.time() - start_time > timeout:
                if cmdDownload.poll() == None:
                    try:
                        cmdDownload.kill()
                    except Exception:
                        pass
                    try:
                        cmdDownload.communicate(timeout=5)
                    except TimeoutExpired:
                        try:
                            cmdDownload.kill()
                        except:
                            pass
                    max_restart -= 1
                    if max_restart < 0:
                        raise Exception("The process is timeout and will not be restarted: "+" ".join(cmd)+"\n")
                    else:
                        cmdDownload = Popen(cmd, stdout=PIPE, stderr=PIPE)
                        ps_proc = psutil.Process(cmdDownload.pid)
                        start_time = time.time()
            else:
                time.sleep(5)
    except psutil.NoSuchProcess:
        # The process has finished
        pass
    
    stdout, stderror = cmdDownload.communicate()
    exitCode = cmdDownload.returncode
    if exitCode != 0:
        raise Exception("This cmd is in error: "+" ".join(cmd)+"\n"+str(stderror.decode("utf-8"))+"\n"+str(stdout.decode("utf-8"))+"\nReturn code: "+str(exitCode)+"\n")
    return stdout, stderror, exitCode

def remove_element_without_bug(list_set, element):
    try:
        list_set.remove(element)
    except:
        pass
    
def extract_ffmpeg_type_dict(filePath):
    import json
    stdout, stderror, exitCode = launch_cmdExt_with_tester([software["ffprobe"], "-v", "error", "-select_streams", "s", "-show_streams", "-of", "json", filePath], 5, 60)
    data_sub_codec = json.loads(stdout.decode("UTF-8"))
    dic_index_data_sub_codec = {}
    for data in data_sub_codec["streams"]:
        dic_index_data_sub_codec[data["index"]] = data
    return dic_index_data_sub_codec

def extract_ffmpeg_type_dict_all(filePath):
    import json
    stdout, stderror, exitCode = launch_cmdExt_with_tester([software["ffprobe"], "-v", "error", "-show_streams", "-of", "json", filePath], 5, 60)
    data_sub_codec = json.loads(stdout.decode("UTF-8"))
    dic_index_data_sub_codec = {}
    for data in data_sub_codec["streams"]:
        dic_index_data_sub_codec[data["index"]] = data
    return dic_index_data_sub_codec

tmpFolder_original = "/tmp"
tmpFolder = "/tmp"
software = {}
core_to_use = 1
default_language_for_undetermine = 'und'
dev = False
special_params = {}
mergeRules = None
sub_type_not_encodable = set(["hdmv_pgs_subtitle","dvd_subtitle","s_hdmv/pgs","pgs"])
sub_type_near_srt = set(["srt","utf-8","utf-16","utf-16le","utf-16be","utf-32","utf-32le","utf-32be","vtt","webvtt","subrip"])
folder_error = "."
group_title_sub = {}
language_to_keep = []
language_to_completely_remove = set()
language_to_try_to_keep = []