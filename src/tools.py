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
    """Return the key/value pairs of one section of an ini file; raise if the section is missing."""
    parser = ConfigParser()
    parser.read(file)

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
    """Return True if the file can be opened for reading."""
    try:
        with open(f):
            return True
    except IOError:
        return False
    
def file_remove(path,file=None):
    """Remove `path`, or `file` inside directory `path`."""
    if file is None:
        os.remove(path)
    else:
        os.remove(os.path.join(path,file))

def make_dirs(d):
    """Create a directory tree; return True if it exists afterwards."""
    try:
        os.makedirs(d,exist_ok=True)
        return os.path.isdir(d)
    except:
        return False
    
def move_dir(Dir,Folder,raise_exception=True):
    """Move a directory; return (ok, error), raising instead when `raise_exception`."""
    try:
        shutil.move(Dir,Folder)
        return True,None
    except Exception as e:
        if raise_exception:
            raise e
        else:
            return False,e
    
def remove_dir(dir_path,printError=True):
    """Remove a directory tree, writing any error to stderr when `printError`."""
    try:
        shutil.rmtree(dir_path)
    except OSError as e:
        if printError:
            sys.stderr.write("Error: %s : %s\n" % (dir_path, e.strerror))

''' Popen functions '''
def launch_cmdExt(cmd):
    """Run a command and return (stdout, stderr, exit_code); raise on a non-zero exit."""
    cmdDownload = Popen(cmd, stdout=PIPE, stderr=PIPE)
    stdout, stderror = cmdDownload.communicate()
    exitCode = cmdDownload.returncode
    if exitCode != 0:
        raise Exception("This cmd is in error: "+" ".join(cmd)+"\n"+str(stderror.decode("utf-8"))+"\n"+str(stdout.decode("utf-8"))+"\nReturn code: "+str(exitCode)+"\n")
    return stdout, stderror, exitCode

def launch_cmdExt_no_test(cmd):
    """Run a command and return (stdout, stderr, exit_code) without checking the exit code."""
    cmdDownload = Popen(cmd, stdout=PIPE, stderr=PIPE)
    stdout, stderror = cmdDownload.communicate()
    exitCode = cmdDownload.returncode
    return stdout, stderror, exitCode

def launch_cmdExt_with_tester(cmd,max_restart=1,timeout=120):
    """Run a command, restarting it when it stalls (zombie or idle CPU) or exceeds `timeout`.

    Returns (stdout, stderr, exit_code); raises after `max_restart` restarts or on a non-zero exit.
    """
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
        pass
    
    stdout, stderror = cmdDownload.communicate(timeout=5)
    exitCode = cmdDownload.returncode
    if exitCode != 0:
        raise Exception("This cmd is in error: "+" ".join(cmd)+"\n"+str(stderror.decode("utf-8"))+"\n"+str(stdout.decode("utf-8"))+"\nReturn code: "+str(exitCode)+"\n")
    return stdout, stderror, exitCode

def launch_cmdExt_with_timeout_reload(cmd,max_restart=1,timeout=120):
    """Run a command with a `timeout`, restarting it up to `max_restart` times.

    Returns (stdout, stderr, exit_code); raises on a non-zero exit.
    """
    unpocessed = True
    while unpocessed:
        cmdDownload = Popen(cmd, stdout=PIPE, stderr=PIPE)
        try:
            stdout, stderror = cmdDownload.communicate(timeout=timeout)
            exitCode = cmdDownload.returncode
            unpocessed = False
        except TimeoutExpired:
            force_kill_subprocess(cmdDownload)
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

def force_kill_subprocess(object_popen,retry=0):
    """Kill a Popen and wait for it, retrying up to 10 times."""
    try:
        object_popen.kill()
        object_popen.wait(timeout=15)
    except TimeoutExpired:
        if retry < 10:
            force_kill_subprocess(object_popen,retry=retry+1)
    except:
        pass

def remove_element_without_bug(list_set, element):
    """Remove `element` from a list or set, ignoring a missing element."""
    try:
        list_set.remove(element)
    except:
        pass
    
def extract_ffmpeg_type_dict(filePath):
    """Return ffprobe's subtitle stream dicts of a file, keyed by stream index."""
    import json
    stdout, stderror, exitCode = launch_cmdExt_with_timeout_reload([software["ffprobe"], "-v", "error", "-select_streams", "s", "-show_streams", "-of", "json", filePath],max_restart=3,timeout=60)
    data_sub_codec = json.loads(stdout.decode("UTF-8"))
    dic_index_data_sub_codec = {}
    for data in data_sub_codec["streams"]:
        dic_index_data_sub_codec[data["index"]] = data
    return dic_index_data_sub_codec

def extract_ffmpeg_type_dict_all(filePath):
    """Return ffprobe's stream dicts of a file, keyed by stream index."""
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
# Subtitle codec classification, spelled in every vocabulary that reaches us:
# ffprobe codec_name, Matroska CodecID and mediainfo Format. Codecs are refused
# by exclusion, so an unknown codec fails loudly instead of being dropped.
# Bitmap codecs are never encoded to text; plain-text codecs are written back as
# srt; everything else is written back as ass, which loses no styling.
sub_type_not_encodable = set([
    "hdmv_pgs_subtitle", "s_hdmv/pgs", "pgs",                    # Blu-ray PGS
    "dvd_subtitle", "s_vobsub", "vobsub",                        # DVD VobSub
    "dvb_subtitle", "s_dvbsub", "dvbsub", "dvb subtitle",        # broadcast DVB
    "xsub", "s_image/xsub",                                      # DivX-era XSUB
])

# Plain-text codecs, written back as srt.
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
    """Return the deployment SHA injected at build time via VMSAM_GIT_COMMIT.

    The runtime image ships no .git directory, so build metadata is the only source.
    """
    return os.environ.get("VMSAM_GIT_COMMIT", "").strip() or "unknown"

mode_test = "test"
mode_production = "production"

def get_execution_mode():
    """Return VMSAM_MODE, read at call time; defaults to `test`.

    An unset or misspelled value never selects the destructive branch.
    """
    return os.environ.get("VMSAM_MODE", mode_test).strip().lower()

logs = []

def dev_num(value, decimals=3):
    """Format one number for a diagnostic line, without ever raising.

    Accepts None, numpy scalars and Decimal; anything unformattable falls back to str().
    """
    if value == None:
        return "n/a"
    try:
        return f"{float(value):.{int(decimals)}f}"
    except (TypeError, ValueError):
        return str(value)


def dev_list(values, decimals=1, limit=12):
    """Format a sequence with `dev_num`, bracketed and truncated to `limit` items. Never raises.

    A non-sequence falls back to `dev_num`.
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

# Port of the internal merge API, bound to 127.0.0.1 only.
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


# Read from the environment because uvicorn worker processes import this module
# fresh and never see the value set in __main__.
def get_dev_env_var():
    """Return the `dev` environment flag (default True)."""
    return _env_flag("dev", True)

''' Verrou d'episode, partage entre la boucle d'integration et le worker de fusion '''
def episode_lock_path(folder_id, episode_number):
    return os.path.join(tmpFolder_original, "locks", f"{folder_id}_{episode_number}.lock")


def acquire_episode_lock(folder_id, episode_number, blocking=True):
    """Take an episode's file lock; return the handle, or None if it is held.

    The integration loop and the merge worker run in separate processes, so the
    lock is a flock, released by the kernel if the holder dies. With
    blocking=False the call returns immediately when the lock is held.
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
    """Release an episode lock; accepts None."""
    if handle == None:
        return
    try:
        fcntl.flock(handle, fcntl.LOCK_UN)
    except OSError:
        pass
    handle.close()


config_file = "config.ini"

def load_merge_runtime_from_env():
    """Rebuild the runtime state a merge needs from the environment and config files.

    A child started with `forkserver` or `spawn` does not inherit the state set in
    __main__, so the internal instance calls this at startup. Idempotent.
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

# Line prefixes that get a UTC timestamp; other lines (read by prefix-based parsers) stay unchanged.
STAMPED_PREFIXES = ("orchestrator: ", "chimeric: ", "scene_anchor: ")


def stamp(message):
    """Insert `utc=<ISO-8601>Z` after a `STAMPED_PREFIXES` prefix; return other lines unchanged."""
    import datetime
    for prefix in STAMPED_PREFIXES:
        if message.startswith(prefix):
            now = datetime.datetime.now(datetime.timezone.utc)
            return (f"{prefix}utc={now.strftime('%Y-%m-%dT%H:%M:%S')}."
                    f"{now.microsecond // 1000:03d}Z {message[len(prefix):]}")
    return message


def log_line(message):
    """Append a line to `logs`, stamped when it is a step line."""
    logs.append(stamp(message))


class decoder_timeout(Exception):
    """An external decoder (ffmpeg, PySceneDetect) ran past its time bound.

    Describes the tool on this host, not the media, so the file can be retried later.
    """

    def __init__(self, tool, seconds, detail=""):
        super().__init__(f"{tool} ran past its {round(seconds, 1)} s bound {detail}".strip())
        self.tool = tool
        self.seconds = seconds


# Time bound of one decode: max(120 s, 1 s per second of media), generous enough for a loaded host.
DECODER_TIMEOUT_BASE_S = 120.0
DECODER_TIMEOUT_PER_MEDIA_S = 1.0


def decoder_timeout_for(media_seconds):
    """Return the timeout in seconds for decoding `media_seconds` of media."""
    return max(DECODER_TIMEOUT_BASE_S, DECODER_TIMEOUT_PER_MEDIA_S * max(0.0, float(media_seconds)))


def dev_log(message):
    """Write one trace line to stderr and `logs`, only when `dev` is set.

    `message` carries its own prefix and trailing newline; the line is stamped.
    """
    if dev:
        message = stamp(message)
        sys.stderr.write(message)
        logs.append(message)

def log_always(message):
    """Write one line to stderr and `logs` regardless of `dev`.

    Reserved for terminal verdicts (DECLINED, FAILED, per-candidate outcome), since stderr is capped.
    """
    message = stamp(message)
    sys.stderr.write(message)
    logs.append(message)

def keep_best_audio_fabricated_status(audio):
    """Return which marker, if any, flags this audio dict as fabricated.

    Checks both the in-process 'fabricated' key and the container tag extra.VMSAM_FABRICATED.
    """
    via_key = bool(audio.get('fabricated'))
    via_extra = bool(audio.get('extra', {}).get('VMSAM_FABRICATED'))
    if via_key and via_extra:
        return "both_keys"
    if via_key:
        return "fabricated_key"
    if via_extra:
        return "extra_vmsam_fabricated"
    return "intact"

def keep_best_audio_fabricated_trace(audio_1, audio_2):
    """Classify a pair of audio dicts by fabricated status, log one trace line, return the outcome.

    Outcomes: side1/side2_fabricated_loses, both_ or neither_fabricated_codec_chain.
    Unlocked: the only caller runs after all other threads are joined.
    """
    side1_loses = ((audio_1.get('fabricated') and not audio_2.get('fabricated'))
                  or (audio_1.get('extra', {}).get('VMSAM_FABRICATED')
                      and not audio_2.get('extra', {}).get('VMSAM_FABRICATED')))
    side2_loses = ((audio_2.get('fabricated') and not audio_1.get('fabricated'))
                  or (audio_2.get('extra', {}).get('VMSAM_FABRICATED')
                      and not audio_1.get('extra', {}).get('VMSAM_FABRICATED')))
    status_1 = keep_best_audio_fabricated_status(audio_1)
    status_2 = keep_best_audio_fabricated_status(audio_2)
    if side1_loses:
        outcome = "side1_fabricated_loses"
    elif side2_loses:
        outcome = "side2_fabricated_loses"
    elif status_1 != "intact" or status_2 != "intact":
        outcome = "both_fabricated_codec_chain"
    else:
        outcome = "neither_fabricated_codec_chain"
    trace_line = (f"keep_best_audio fabricated_check outcome={outcome} "
                  f"side1_marker={status_1} side2_marker={status_2}")
    if outcome in ("side1_fabricated_loses", "side2_fabricated_loses"):
        dropped, kept = (audio_1, audio_2) if outcome == "side1_fabricated_loses" else (audio_2, audio_1)
        dropped_side = "side1" if outcome == "side1_fabricated_loses" else "side2"
        trace_line += (f" dropped={dropped_side} dropped_lang={dropped.get('Language')} "
                       f"dropped_format={dropped.get('Format')} dropped_stream={dropped.get('StreamOrder')} "
                       f"kept_lang={kept.get('Language')} kept_format={kept.get('Format')} "
                       f"kept_stream={kept.get('StreamOrder')} "
                       f"effect=dropped_track_will_not_be_in_the_delivered_file")
    logs.append(trace_line + "\n")
    return outcome


# Variant of `launch_cmdExt_with_timeout_reload` with the same signature and exceptions, that:
#   - relaunches only once per timeout and closes the killed process's pipes;
#   - adds the tool's verbose flag (see `verbose_cmd_shadow`) so a failure carries its error;
#   - on timeout or non-zero exit, logs the child's state, stderr tail and named files.
verbose_external_tools = True
CHILD_TAIL_LINES = 40


def verbose_cmd_shadow(cmd):
    """Return `cmd` with its tool's verbose flag added (ffmpeg/ffprobe, mkvmerge, mkvextract, mkvpropedit).

    mkvmerge identification commands are left unchanged since their JSON goes to stdout.
    """
    if not verbose_external_tools or not len(cmd):
        return list(cmd)
    tool = os.path.basename(str(cmd[0]))
    cmd = list(cmd)
    if tool in ("ffmpeg", "ffprobe"):
        rest = []
        i = 1
        while i < len(cmd):
            if cmd[i] in ("-v", "-loglevel") and i + 1 < len(cmd):
                i += 2  # replace, never double
                continue
            rest.append(cmd[i])
            i += 1
        head = [cmd[0], "-loglevel", "verbose"]
        if "-hide_banner" not in rest:
            head.append("-hide_banner")
        return head + rest
    if tool == "mkvmerge":
        if any(a in ("-i", "-J", "--identify", "-v", "--verbose") for a in cmd[1:]):
            return cmd
        return [cmd[0], "-v"] + cmd[1:]
    if tool in ("mkvextract", "mkvpropedit"):
        if "--verbose" in cmd[1:] or "-v" in cmd[1:]:
            return cmd
        return [cmd[0], "--verbose"] + cmd[1:]
    return cmd


def read_available_shadow(stream):
    """Read whatever is waiting in a pipe without blocking."""
    if stream == None:
        return b""
    chunks = []
    try:
        fd = stream.fileno()
        os.set_blocking(fd, False)
        while True:
            try:
                chunk = os.read(fd, 65536)
            except BlockingIOError:
                break
            if not chunk:
                break
            chunks.append(chunk)
    except (OSError, ValueError):
        pass
    return b"".join(chunks)


def child_proc_state_shadow(pid):
    """Return (state, wchan) from /proc, or ("gone", "gone") if the process no longer exists."""
    try:
        with open(f"/proc/{pid}/stat") as stat:
            state = stat.read().rsplit(")", 1)[1].split()[0]
    except OSError:
        return "gone", "gone"
    try:
        with open(f"/proc/{pid}/wchan") as wchan:
            channel = wchan.read().strip() or "0"
    except OSError:
        channel = "unreadable"
    return state, channel


def files_named_shadow(lines, cmd):
    """Size and mtime of the existing files named by the stderr tail or by argv."""
    import re
    import datetime
    candidates = []
    for line in lines:
        candidates += re.findall(r"'([^']+)'", line) + re.findall(r'"([^"]+)"', line)
        candidates += re.findall(r"(/[^\s'\":,]+)", line)
    candidates += [str(a) for a in cmd[1:]]
    described = []
    for path in candidates:
        if path in [d[0] for d in described] or not os.path.isfile(path):
            continue
        try:
            info = os.stat(path)
        except OSError:
            continue
        mtime = datetime.datetime.fromtimestamp(info.st_mtime, datetime.timezone.utc)
        described.append((path, info.st_size, mtime.strftime("%Y-%m-%dT%H:%M:%S.%fZ")))
        if len(described) >= 10:
            break
    return described


def log_child_tail_shadow(event, cmd, pid, elapsed, stderror, stdout, returncode=None):
    """Log the child's state, argv, stderr tail and named files. Never raises."""
    try:
        log_child_tail_shadow_unguarded(event, cmd, pid, elapsed, stderror, stdout, returncode)
    except Exception as error:
        log_always(f"tools: child {event} pid={pid} diagnostic failed {type(error).__name__}: {error}\n")


def log_child_tail_shadow_unguarded(event, cmd, pid, elapsed, stderror, stdout, returncode):
    state, channel = child_proc_state_shadow(pid)
    lines = stderror.decode("utf-8", errors="replace").splitlines()
    tail = lines[-CHILD_TAIL_LINES:]
    out = [f"tools: child {event} pid={pid} elapsed_s={round(elapsed, 3)} state={state} "
           f"wchan={channel}" + ("" if returncode == None else f" returncode={returncode}")
           + f" stderr_bytes={len(stderror)} stdout_bytes={len(stdout)}",
           f"tools: child argv={cmd!r}"]
    out += [f"tools: child stderr[{len(lines) - len(tail) + n}] {line}" for n, line in enumerate(tail)]
    for path, size, mtime in files_named_shadow(tail, cmd):
        out.append(f"tools: child file {path!r} size={size} mtime={mtime}")
    log_always("\n".join(out) + "\n")


def launch_cmdExt_with_timeout_reload_shadow(cmd,max_restart=1,timeout=120):
    """Run a command like `launch_cmdExt_with_timeout_reload`, with verbose flags and diagnostics.

    Returns (stdout, stderr, exit_code); raises after `max_restart` timeouts or on a non-zero exit.
    """
    cmd = verbose_cmd_shadow(cmd)
    unpocessed = True
    while unpocessed:
        cmdDownload = Popen(cmd, stdout=PIPE, stderr=PIPE)
        began = time.monotonic()
        try:
            stdout, stderror = cmdDownload.communicate(timeout=timeout)
            exitCode = cmdDownload.returncode
            unpocessed = False
        except TimeoutExpired as expired:
            partial_err = (expired.stderr or b"") + read_available_shadow(cmdDownload.stderr)
            partial_out = (expired.stdout or b"") + read_available_shadow(cmdDownload.stdout)
            log_child_tail_shadow("timeout_before_kill", cmd, cmdDownload.pid,
                                  time.monotonic() - began, partial_err, partial_out)
            force_kill_subprocess(cmdDownload)
            partial_err += read_available_shadow(cmdDownload.stderr)
            partial_out += read_available_shadow(cmdDownload.stdout)
            log_child_tail_shadow("timeout_after_kill", cmd, cmdDownload.pid,
                                  time.monotonic() - began, partial_err, partial_out,
                                  cmdDownload.returncode)
            close_subprocess_pipes_shadow(cmdDownload)
            max_restart -= 1
            if max_restart < 0:
                raise Exception(f"The process is timeout and will not be restarted:{cmd}\n")
            else:
                if dev:
                    sys.stderr.write(f"The process is timeout and will be restarted:{cmd}\n")

    if exitCode != 0:
        log_child_tail_shadow("nonzero_exit", cmd, cmdDownload.pid, time.monotonic() - began,
                              stderror, stdout, exitCode)
        raise Exception("This cmd is in error: "+" ".join(cmd)+"\n"+str(stderror.decode("utf-8"))+"\n"+str(stdout.decode("utf-8"))+"\nReturn code: "+str(exitCode)+"\n")
    return stdout, stderror, exitCode


def close_subprocess_pipes_shadow(object_popen):
    """Close the stdout/stderr pipes of a killed Popen."""
    for stream in (object_popen.stdout, object_popen.stderr):
        try:
            if stream != None:
                stream.close()
        except OSError:
            pass

"""
END: AGENT modification
"""