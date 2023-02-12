'''
Created on 8 Jan 2023

@author: studyfranco
'''

from tools import make_dirs,move_dir

import argparse
import os
import shutil
from re import match,search,MULTILINE
from subprocess import Popen, PIPE
from sys import stderr
from time import sleep

def launch_cmdExt(cmd):
    cmdDownload = Popen(cmd, stdout=PIPE, stderr=PIPE)
    stdout, stderror = cmdDownload.communicate()
    exitCode = cmdDownload.returncode
    return stdout, stderror, exitCode

def process_files(cmd_use_to_process,folder_path,folder_path_for_error,folder_name):
    stdout, stderror, exit_code = launch_cmdExt(cmd_use_to_process)
    if exit_code != 0:
        generate_error_folder(folder_path_for_error)
        with open(os.path.join(folder_path,"log.error"),"w") as log:
            log.write(stderror.decode("utf-8"))
        stderr.write(stderror.decode("utf-8"))
        stderr.write("\n")
        with open(os.path.join(folder_path,"log.out"),"w") as log:
            log.write(stdout.decode("utf-8"))
        move_dir(folder_path,folder_path_for_error)
    else:
        if search(r'\[.+\] not compatible with the others videos',stderror.decode("utf-8"), MULTILINE) != None:
            list_files_not_compatible_to_move = search(r'\[(.+)\] not compatible with the others videos',stderror.decode("utf-8"), MULTILINE).group(1).split(", ")
            generate_error_folder(os.path.join(folder_path_for_error,folder_name+"_not_compatible_files"))
            for file in list_files_not_compatible_to_move:
                shutil.move(file[1:-1],os.path.join(folder_path_for_error,folder_name+"_not_compatible_files"))
        with open(os.path.join(args.error,"log.error"),"a") as log:
            log.write("\n"+"#"*20+"\n"+folder_path+"\n"+"#"*20+"\n")
            log.write(stderror.decode("utf-8")+"\n")
        with open(os.path.join(args.error,"log.log"),"a") as log:
            log.write("\n"+"#"*20+"\n"+folder_path+"\n"+"#"*20+"\n")
            log.write(stdout.decode("utf-8")+"\n")
        shutil.rmtree(folder_path)
    
def generate_error_folder(folder_path_for_error):
    if (not make_dirs(folder_path_for_error)):
        raise Exception(f"Impossible to create {folder_path_for_error}")
    
def process_files_in_folder(folder,original_folder,out_folder,folder_path_for_error):
    folder_path = os.path.join(original_folder,folder)
    if match(r'\S+.*\s*\[grouping\]\s*.*',folder) != None:
        try:
            name_folder_clean = " ".join([search(r'(\S+.*)\s*\[grouping\]\s*(.*)',folder).group(1),search(r'(\S+)\s*\[grouping\]\s*(.*)',folder).group(2)])
            if len(name_folder_clean) == 0:
                raise Exception("")
        except:
            stderr.write(f'{folder} is not a good name for a folder')
            stderr.write("\n")
            generate_error_folder(folder_path_for_error)
            move_dir(folder_path,folder_path_for_error)
        else:
            for name_folder in [ name for name in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, name)) ]:
                process_files_in_folder(name_folder,folder_path,os.path.join(out_folder,name_folder_clean),os.path.join(folder_path_for_error,folder))
                shutil.rmtree(folder_path)
    else:
        if (not make_dirs(out_folder)):
            stderr.write(f"Impossible to create {out_folder}")
            stderr.write("\n")
            generate_error_folder(folder_path_for_error)
            move_dir(folder_path,folder_path_for_error)
        else:
            global cmd_use_to_process
            cmd_use_to_process = base_cmd_use_to_process.copy()
            cmd_use_to_process.extend(["-o", out_folder])
            list_files_to_process = []
            for file in os.listdir(folder_path):
                if file.endswith(".mkv") or file.endswith(".mp4") or file.endswith(".avi"):
                    list_files_to_process.append(os.path.join(folder_path, file))
                elif match(r'param.json',file) != None:
                    cmd_use_to_process.extend(["--param", os.path.join(folder_path, file)])
                elif os.path.isdir(os.path.join(folder_path, file)):
                    generate_error_folder(os.path.join(folder_path_for_error,folder+"_not_process_files"))
                    move_dir(os.path.join(folder_path, file), os.path.join(folder_path_for_error,folder+"_not_process_files"))
                else:
                    generate_error_folder(os.path.join(folder_path_for_error,folder+"_not_process_files"))
                    shutil.move(os.path.join(folder_path, file),os.path.join(folder_path_for_error,folder+"_not_process_files"))
                    
            if len(list_files_to_process) > 1:
                cmd_use_to_process.extend(["-f", ",".join(list_files_to_process)])
                if match(r'\S*.*\s*\[merge\]\s*.*',folder) != None:
                    cmd_use_to_process.append("--noSync")
                    process_files(cmd_use_to_process,folder_path,folder_path_for_error,folder)
                else:
                    process_files(cmd_use_to_process,folder_path,folder_path_for_error,folder)
            else:
                with open(os.path.join(folder_path,"log.error"),"w") as log:
                    log.write(f"No goods files in {folder_path}")
                stderr.write(f"No goods files in {folder_path}")
                stderr.write("\n")
                generate_error_folder(folder_path_for_error)
                move_dir(folder_path,folder_path_for_error)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='This script is the wrapper to process mkv,mp4 file to generate best file', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--folder", metavar='folder', type=str,
                       required=True, help="If All files are in the same folder, in this option give the path of the folder who contain files to merge and don't write it for all files")
    parser.add_argument("-o","--out", metavar='outdir', type=str, default=".", help="Folder where send new files")
    parser.add_argument("-e","--error", metavar='error', type=str, default=".", help="Folder where send files with errors")
    parser.add_argument("--tmp", metavar='tmpdir', type=str,
                        default="/tmp", help="Folder where send temporar files")
    parser.add_argument("--pwd", metavar='pwd', type=str,
                        default=".", help="Path to the merge software")
    parser.add_argument("-c","--core", metavar='core', type=int, default=1, help="number of core the merge software can use")
    parser.add_argument("-w","--wait", metavar='wait', type=int, default=300, help="Time in second between folder check")
    parser.add_argument("--dev", dest='dev', default=False, action='store_true', help="Print more errors and write all logs")
    args = parser.parse_args()
    if (not os.access(args.out, os.W_OK)):
        raise Exception(f"{args.out} not writable")
    if (not os.access(args.error, os.W_OK)):
        raise Exception(f"{args.error} not writable")
    if (not os.access(args.folder, os.W_OK)):
        raise Exception(f"{args.folder} not writable")
    base_cmd_use_to_process = ["python3", os.path.join(args.pwd,"mergeVideo.py"), "--pwd", args.pwd, "-c", str(args.core), "--tmp", args.tmp]
    if args.dev:
        base_cmd_use_to_process.append("--dev")
    while True:
        folder_to_clean = [ name_folder for name_folder in os.listdir(args.folder) if os.path.isdir(os.path.join(args.folder, name_folder)) ]
        if len(folder_to_clean):
            for name_folder in folder_to_clean:
                process_files_in_folder(name_folder,args.folder,args.out,args.error)
        else:
            sleep(args.wait)
            