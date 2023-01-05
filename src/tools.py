'''
Created on 23 Apr 2022

@author: francois
'''
import os
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

def remove_element_without_bug(list_set, element):
    try:
        list_set.remove(element)
    except:
        pass

''' Files functions '''
def file_exists(f):
    try:
        with open(f):
            return True
    except IOError:
        return False
    
def file_remove(path,file):
    os.remove(os.path.join(path,file))

''' Popen functions '''
def launch_cmdExt(cmd):
    cmdDownload = Popen(cmd, stdout=PIPE, stderr=PIPE)
    stdout, stderror = cmdDownload.communicate()
    exitCode = cmdDownload.returncode
    if exitCode != 0:
        raise Exception(str(stderror))
    return stdout, stderror, exitCode

def make_dirs(d):
    try:
        os.makedirs(d)
        return True
    except:
        return False

tmpFolder = "/tmp"
software = {}
core_to_use = 1