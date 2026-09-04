'''
Created on 23 Apr 2022

@author: studyfranco
'''
import fcntl
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
    
def file_remove(path,file=None):
    if file is None:
        os.remove(path)
    else:
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
    exitCode = 5555
    global dev
    try:
        ps_proc = psutil.Process(cmdDownload.pid)
        start_time = time.time()
        
        while cmdDownload.poll() == None and exitCode != 0:
            time.sleep(10)
            if cmdDownload.poll() == None and (ps_proc.status() == psutil.STATUS_ZOMBIE or ps_proc.cpu_percent(interval=1.0) < 0.05):
                if ps_proc.cpu_percent(interval=2.0) < 0.05 and cmdDownload.poll() == None:
                    stdout = None
                    stderror = None
                    try:
                        stdout, stderror = cmdDownload.communicate(timeout=5)
                        exitCode = cmdDownload.returncode
                    except TimeoutExpired:
                        try:
                            cmdDownload.kill()
                        except:
                            pass
                    
                    if exitCode != 0:
                        max_restart -= 1
                        if max_restart < 0:
                            raise Exception(f"The process is zombie and cannot be restarted:{cmd}\n{stderror}\n{stdout}\n")
                        else:
                            if dev:
                                sys.stderr.write("The process is zombie and will be restarted: "+" ".join(cmd)+"\n")
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
                        if dev:
                            sys.stderr.write("The process is timeout and will be restarted: "+" ".join(cmd)+"\n")
                        cmdDownload = Popen(cmd, stdout=PIPE, stderr=PIPE)
                        ps_proc = psutil.Process(cmdDownload.pid)
                        start_time = time.time()
            else:
                time.sleep(5)
    except psutil.NoSuchProcess:
        # The process has finished
        pass
    
    stdout, stderror = cmdDownload.communicate(timeout=5)
    exitCode = cmdDownload.returncode
    if exitCode != 0:
        raise Exception("This cmd is in error: "+" ".join(cmd)+"\n"+str(stderror.decode("utf-8"))+"\n"+str(stdout.decode("utf-8"))+"\nReturn code: "+str(exitCode)+"\n")
    return stdout, stderror, exitCode

def launch_cmdExt_with_timeout_reload(cmd,max_restart=1,timeout=120):
    unpocessed = True
    while unpocessed:
        cmdDownload = Popen(cmd, stdout=PIPE, stderr=PIPE)
        try:
            stdout, stderror = cmdDownload.communicate(timeout=timeout)
            exitCode = cmdDownload.returncode
            unpocessed = False
        except TimeoutExpired:
            try:
                cmdDownload.kill()
            except:
                pass
            max_restart -= 1
            if max_restart < 0:
                raise Exception(f"The process is timeout and will not be restarted:{cmd}\n")
            else:
                if dev:
                    sys.stderr.write(f"The process is timeout and will be restarted:{cmd}\n")
                cmdDownload = Popen(cmd, stdout=PIPE, stderr=PIPE)
    
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
    stdout, stderror, exitCode = launch_cmdExt_with_timeout_reload([software["ffprobe"], "-v", "error", "-select_streams", "s", "-show_streams", "-of", "json", filePath],max_restart=3,timeout=60)
    data_sub_codec = json.loads(stdout.decode("UTF-8"))
    dic_index_data_sub_codec = {}
    for data in data_sub_codec["streams"]:
        dic_index_data_sub_codec[data["index"]] = data
    return dic_index_data_sub_codec

def extract_ffmpeg_type_dict_all(filePath):
    import json
    stdout, stderror, exitCode = launch_cmdExt_with_timeout_reload([software["ffprobe"], "-v", "error", "-show_streams", "-of", "json", filePath],max_restart=3,timeout=60)
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
# The single classification for subtitle codecs. Both names are compared against
# TWO vocabularies -- ffprobe's `codec_name` and mediainfo's `Format` -- plus the
# Matroska CodecIDs that reach us through the container, so each entry is spelled
# in whichever of the three actually appears.
#
# Refuse by exclusion, never by allow-list: a codec missing from an allow-list is
# dropped in silence and reported as an image, while a codec missing from an
# exclusion list is attempted and fails loudly on a real defect. Anything not in
# `sub_type_not_encodable` is written back as srt when it is in
# `sub_type_near_srt`, and as ass otherwise.
#
# WHEN IN DOUBT, ASS. Ass expresses everything srt can plus styling, so calling a
# plain format styled costs a slightly heavier container, while calling a styled
# format plain destroys its styling with no error. That asymmetry decides every
# uncertain case, which is why ttml, sami, jacosub, realtext, microdvd, stl and
# mov_text are absent below: each can carry style and none is worth losing on a
# guess.
# BITMAP -- never encoded to text. One line per family, spelled in every vocabulary
# that reaches us: ffprobe codec_name, Matroska CodecID, mediainfo Format.
sub_type_not_encodable = set([
    "hdmv_pgs_subtitle", "s_hdmv/pgs", "pgs",                    # Blu-ray PGS
    "dvd_subtitle", "s_vobsub", "vobsub",                        # DVD VobSub
    "dvb_subtitle", "s_dvbsub", "dvbsub", "dvb subtitle",        # broadcast DVB
    "xsub", "s_image/xsub",                                      # DivX-era XSUB
])

# PLAIN TEXT -- written back as srt, because srt loses nothing they can express.
# Everything not listed here and not bitmap is written back as ass.
sub_type_near_srt = set([
    "subrip", "srt", "s_text/utf8",                              # SubRip
    "utf-8", "utf-16", "utf-16le", "utf-16be",                   # mediainfo Format
    "utf-32", "utf-32le", "utf-32be",
    "webvtt", "vtt", "s_text/webvtt",                            # WebVTT
    "text",                                                      # raw timed text
    "mpl2", "pjs", "subviewer", "subviewer1", "vplayer",         # plain, no styling
])
to_convert_ffmpeg_type = {
    "webvtt": ["webvtt","srt"],
    "s_text/webvtt": ["webvtt","srt"]
}
folder_error = "."
group_title_sub = {}
language_to_keep = []
language_to_completely_remove = set()
language_to_try_to_keep = []

def get_git_commit():
    """Deployment SHA, injected at build time via VMSAM_GIT_COMMIT.

    The runtime image ships no .git directory, so build metadata is the only
    source. Lue a chaque appel: une variable d'environnement ne change pas en
    cours de process, et le cache ne payait pas son global.
    """
    return os.environ.get("VMSAM_GIT_COMMIT", "").strip() or "unknown"

mode_test = "test"
mode_production = "production"

def get_execution_mode():
    """Read VMSAM_MODE at call time so switching mode needs no code change.

    Defaults to `test`: an unset or misspelled value must never silently take
    the destructive branch.
    """
    return os.environ.get("VMSAM_MODE", mode_test).strip().lower()

logs = []

def dev_num(value, decimals=3):
    """Format one number for a `tools.dev` diagnostic line, without ever raising.

    Diagnostics run inside merges, so a logging path that can raise is a defect,
    not a cosmetic problem. The values these lines carry are frequently `None`
    (a measurement that did not happen), a numpy scalar, or a `Decimal`, and a
    plain f-string format spec raises `TypeError` on the first of those. Every
    input yields a string here; nothing propagates.
    """
    if value == None:
        return "n/a"
    try:
        return f"{float(value):.{int(decimals)}f}"
    except (TypeError, ValueError):
        return str(value)


def dev_list(values, decimals=1, limit=12):
    """`dev_num` over a sequence, bracketed and truncated. Never raises.

    Merges are chatty and a full per-cut vector can run to dozens of entries, so
    the tail is summarised as a count rather than printed. A non-sequence falls
    back to `dev_num`, which is what makes this safe to call on whatever a
    measurement returned.
    """
    if isinstance(values, (str, bytes)):
        return str(values)
    try:
        items = list(values)
    except TypeError:
        return dev_num(values, decimals)
    shown = ", ".join(dev_num(item, decimals) for item in items[:limit])
    if len(items) > limit:
        shown = shown + f", +{len(items) - limit} more"
    return "[" + shown + "]"

# Port de l'API interne de fusion, liee a 127.0.0.1 et jamais exposee.
internal_api_port = 42085

def _env_flag(name, default=False):
    """Read a boolean from the environment. Absent or unparseable keeps `default`."""
    raw = os.environ.get(name)
    if raw == None:
        return default
    raw = raw.strip().lower()
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("0", "false", "no", "off", ""):
        return False
    return default


# Defaulted from the environment, not just from --dev, because uvicorn's public
# instance runs with workers=5 and those are SEPARATE processes: they import
# this module fresh and never see an assignment made in __main__. That is why
# GET /health reported dev=False while the internal single-worker instance --
# the one that actually runs merges -- had it True. --dev still forces it on.
def get_dev_env_var():
    return _env_flag("dev", True)

''' Verrou d'episode, partage entre la boucle d'integration et le worker de fusion '''
def episode_lock_path(folder_id, episode_number):
    return os.path.join(tmpFolder_original, "locks", f"{folder_id}_{episode_number}.lock")


def acquire_episode_lock(folder_id, episode_number, blocking=True):
    """Prend le verrou d'un episode. Retourne le handle, ou None si occupe.

    Les deux ecrivains d'un episode sont dans des process distincts -- la boucle
    d'integration et le worker de fusion -- donc le verrou doit etre au niveau
    du systeme: flock sur un fichier, libere par le noyau si le process meurt,
    ce qu'un drapeau en base ou en memoire ne garantit pas.

    blocking=False rend la main immediatement quand le verrou est tenu: c'est ce
    que veut la boucle d'integration, qui repassera sur cet episode au prochain
    tour plutot que d'attendre la fin d'une fusion.
    """
    if not make_dirs(os.path.dirname(episode_lock_path(folder_id, episode_number))):
        return None
    handle = open(episode_lock_path(folder_id, episode_number), "w")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX if blocking else (fcntl.LOCK_EX | fcntl.LOCK_NB))
    except OSError:
        handle.close()
        return None
    return handle


def release_episode_lock(handle):
    """Libere un verrou. Tolere None, pour que l'appelant n'ait pas a tester."""
    if handle == None:
        return
    try:
        fcntl.flock(handle, fcntl.LOCK_UN)
    except OSError:
        pass
    handle.close()


config_file = "config.ini"

def load_merge_runtime_from_env():
    """Reconstruit l'etat runtime necessaire a un merge, depuis l'environnement.

    __main__ construit cet etat puis lance l'instance interne dans un Process.
    Tant que la methode de demarrage etait `fork`, l'enfant en heritait
    implicitement. Python 3.14 est passe a `forkserver` par defaut: un enfant n'y
    voit plus que les valeurs par defaut de ce module, et l'echec est SILENCIEUX
    -- tmpFolder_original retombe a "/tmp", donc les verrous d'episode des deux
    process partent dans deux arbres differents et n'excluent plus rien, pendant
    que software={} fait echouer tout appel a ffmpeg.

    Appeler ceci au demarrage de l'instance interne la rend independante de la
    methode de demarrage. C'est idempotent: sous `fork` les valeurs sont deja
    correctes et on relit les memes fichiers.
    """
    global tmpFolder_original, tmpFolder, core_to_use, folder_error, software
    global mergeRules, group_title_sub, language_to_keep
    global language_to_completely_remove, language_to_try_to_keep, special_params

    tmp_original = os.environ.get("VMSAM_TMP_FOLDER_ORIGINAL", "").strip()
    if len(tmp_original):
        tmpFolder_original = tmp_original
        tmpFolder = os.path.dirname(tmp_original) or tmpFolder
    core = os.environ.get("VMSAM_CORE_TO_USE", "").strip()
    if core.isdigit():
        core_to_use = max(1, int(core))
    error_folder = os.environ.get("VMSAM_FOLDER_ERROR", "").strip()
    if len(error_folder):
        folder_error = error_folder

    current_config = os.environ.get("VMSAM_CONFIG", "").strip() or config_file
    software = config_loader(current_config, "software")
    mergeRules = config_loader(current_config, "mergerules")

    import json
    with open("titles_subs_group.json") as titles_subs_group_file:
        group_title_sub = json.load(titles_subs_group_file)
    with open("config.json") as configuration_file:
        configuration = json.load(configuration_file)
    language_to_keep = configuration["language_to_keep"]
    language_to_completely_remove = set(configuration["language_to_completely_remove"])
    language_to_try_to_keep = configuration["language_to_try_to_keep"]

    special_params = {"change_all_und":True, "remove_commentary":True,
                      "remove_descriptive":True, "forced_best_video_contain":False}

"""
BEGIN: AGENT modification ok
"""

"""
END: AGENT modification
"""