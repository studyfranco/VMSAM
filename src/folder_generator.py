'''
Created on 22 Jan 2023

@author: franco
'''

import argparse
import os

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
    args = parser.parse_args()
    if (not os.access(args.out, os.W_OK)):
        raise Exception(f"{args.out} not writable")
    if args.noSync:
        for i in range(0,args.number):
            make_dirs(os.path.join(args.out,str(i+1)+" [merge]"))
    else:
        for i in range(0,args.number):
            make_dirs(os.path.join(args.out,str(i+1)))