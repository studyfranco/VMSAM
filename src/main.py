import argparse
from datetime import datetime
from multiprocessing import Pool
from os import path,chdir
import traceback
import tools

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
                        default="config.ini", help="Path to the config file, by default use the config in the software folder. This config is for configure the path to your softwares")
    parser.add_argument("--param", metavar='param', type=str,
                       default=None, help="Give the path to a special file for your merge.")
    parser.add_argument("--pwd", metavar='pwd', type=str,
                        default=".", help="Path to the software, put it if you use the folder from another folder")
    parser.add_argument("--noSync", dest='noSync', default=False, action='store_true', help="If you don't want research a audio sync between files")
    parser.add_argument("--dev", dest='dev', default=False, action='store_true', help="Print more errors and write all logs")
    args = parser.parse_args()
    
    chdir(args.pwd)
    tools.tmpFolder = path.join(args.tmp,"VMSAM_"+str(datetime.now().strftime("%Y-%m-%d_%H:%M:%S")))
    tools.dev = args.dev
    
    try:
        tools.software = tools.config_loader(args.config, "software")
        if (not tools.make_dirs(tools.tmpFolder)):
            raise Exception("Impossible to create the temporar dir")

        if args.core > 1:
            tools.core_to_use = args.core-1
        else:
            tools.core_to_use = 1

        import mergeVideo
        import video
        import json
        if args.param != None:
            with open(args.param) as param_file:
                tools.special_params = json.load(param_file)
            tools.default_language_for_undetermine = tools.special_params["default_language_und"]
            if "model_path" in tools.special_params and tools.special_params['model_path'] != "" and tools.special_params['model_path'] != None:
                video.path_to_livmaf_model = ":model_path="+tools.special_params['model_path']
            video.number_cut = tools.special_params["number_cut"]
            mergeVideo.cut_file_to_get_delay_second_method = tools.special_params["second_cut_lenght"]
        else:
            tools.special_params = {"change_all_und":False, "original_language":"", "remove_commentary":False, "forced_best_video":"", "forced_best_video_contain":False}
        
        tools.mergeRules = tools.config_loader(args.config,"mergerules")
        
        with open("titles_subs_group.json") as titles_subs_group_file:
            tools.group_title_sub = json.load(titles_subs_group_file)

        video.ffmpeg_pool_audio_convert = Pool(processes=tools.core_to_use)
        video.ffmpeg_pool_big_job = Pool(processes=1)

        mergeVideo.merge_videos(set(args.file.split(",")), args.out, (not args.noSync), args.folder)
        tools.remove_dir(tools.tmpFolder)
    except:
        tools.remove_dir(tools.tmpFolder)
        traceback.print_exc()
        exit(1)
    exit(0)