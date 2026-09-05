#!/usr/bin/env python3
"""Apply the capture hook to a checkout's mergeVideo.py. BUILD-TIME ONLY, NEVER COMMITTED.

Idempotent: refuses to double-apply. Verifies the result COMPILES before writing --
a patch that lands a syntax error into an image build is worse than one that refuses.
"""
import py_compile, re, shutil, sys, tempfile, os

MARK = "# --- ci fixture capture hook (lab, not committed) ---"

SITE1_OLD = ("    remove_not_compatible_video(list_not_compatible_video,"
             "dict_file_path_obj,compareObjs[0])")
SITE1_NEW = (MARK + "\n"
             "    import fixture_capture as _fc\n"
             "    _fc.capture('site1_compareObjs0', list_not_compatible_video,\n"
             "                dict_file_path_obj,\n"
             "                compareObjs[0] if compareObjs else None,\n"
             "                compare_objs=compareObjs)\n"
             + SITE1_OLD)

SITE2_OLD = ("        remove_not_compatible_video(list_not_compatible_video,"
             "dict_file_path_obj,dict_file_path_obj[forced_best_video])")
SITE2_NEW = (MARK + "\n"
             "        import fixture_capture as _fc\n"
             "        _fc.capture('site2_forced_best_video', list_not_compatible_video,\n"
             "                    dict_file_path_obj,\n"
             "                    dict_file_path_obj.get(forced_best_video))\n"
             + SITE2_OLD)


def apply(path):
    src = open(path).read()
    if MARK in src:
        return "ALREADY APPLIED -- refusing to double-apply"
    for old in (SITE1_OLD, SITE2_OLD):
        if src.count(old) != 1:
            return f"REFUSING: expected exactly 1 occurrence, found {src.count(old)}"
    out = src.replace(SITE1_OLD, SITE1_NEW, 1).replace(SITE2_OLD, SITE2_NEW, 1)
    fd, tmp = tempfile.mkstemp(suffix='.py'); os.close(fd)
    open(tmp, 'w').write(out)
    try:
        py_compile.compile(tmp, doraise=True)
    except py_compile.PyCompileError as e:
        os.unlink(tmp)
        return f"REFUSING: patched file does not compile -- {str(e)[:90]}"
    shutil.copy(path, path + '.pre-hook')
    shutil.move(tmp, path)
    return "APPLIED to both call sites; original kept at .pre-hook"


if __name__ == '__main__':
    print("  " + apply(sys.argv[1]))
