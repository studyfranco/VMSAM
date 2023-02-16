'''
Created on 22 Jan 2023

@author: franco
'''

import argparse
import os
import shutil

def make_dirs(d):
    try:
        os.makedirs(d,exist_ok=True)
        return os.path.isdir(d)
    except:
        return False

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='This script help you to generate folder for your merge', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-o","--out", metavar='outdir', type=str, default=".", help="Folder where send new files")
    parser.add_argument("--noSync", dest='noSync', default=False, action='store_true', help="If you don't want research a audio sync between files")
    parser.add_argument("-n","--number", metavar='number', type=int, default=1, help="number of folder you want create")
    parser.add_argument("--param", metavar='param', type=str, default=None, help="Give the path to a special file for your merge. It prepare it")
    parser.add_argument("--begin", metavar='begin', type=str, default="", help="Folder begin name")
    args = parser.parse_args()
    if (not os.access(args.out, os.W_OK)):
        raise Exception(f"{args.out} not writable")
    copy_param = args.param != None
    if args.noSync:
        for i in range(0,args.number):
            new_folder = os.path.join(args.out,args.begin+str(i+1)+" [merge]")
            make_dirs(new_folder)
            if copy_param:
                shutil.copyfile(args.param, os.path.join(new_folder,"param.json"))
    else:
        for i in range(0,args.number):
            new_folder = os.path.join(args.out,args.begin+str(i+1))
            make_dirs(new_folder)
            if copy_param:
                shutil.copyfile(args.param, os.path.join(new_folder,"param.json"))