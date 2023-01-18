'''
Created on 23 Apr 2022

@author: francois
'''
import os
import shutil
import sys
from subprocess import Popen, PIPE
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
        raise Exception("This cmd is in error: "+" ".join(cmd)+"\n"+str(stderror.decode("utf-8")))
    return stdout, stderror, exitCode

def remove_element_without_bug(list_set, element):
    try:
        list_set.remove(element)
    except:
        pass

tmpFolder = "/tmp"
software = {}
core_to_use = 1
default_language_for_undetermine = 'und'