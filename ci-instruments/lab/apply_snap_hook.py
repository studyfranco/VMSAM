#!/usr/bin/env python3
"""Wrap adjust_delay_to_frame so every call is captured. BUILD-TIME ONLY, NEVER COMMITTED.

RENAME-AND-WRAP rather than editing the body: the method has FOUR return statements
(two of them unreachable, inside triple-quoted AGENT-modification blocks). Editing
each return would be four chances to change behaviour; wrapping is one, and the
original is called unmodified so the CAPTURED value is the SHIPPED value.

Refuses on mismatch, refuses to double-apply, and verifies the result compiles.
"""
import py_compile, shutil, sys, tempfile, os

MARK = "# --- ci snap capture hook (lab, not committed) ---"
ORIG = "    def adjust_delay_to_frame(self,delay):"
NEW = (MARK + "\n"
       "    def adjust_delay_to_frame(self, delay):\n"
       "        import snap_capture as _sc\n"
       "        _line = _sc.caller_line()\n"
       "        _out = self._ci_orig_adjust_delay_to_frame(delay)\n"
       "        _sc.record(self, delay, _out, _line)\n"
       "        return _out\n"
       "\n"
       "    def _ci_orig_adjust_delay_to_frame(self,delay):")



# --- dev-3's selection-gate column -------------------------------------------
# WHY A SECOND HOOK: the values dev-3 needs are locals of test_if_constant_good_delay,
# not of adjust_delay_to_frame, so the snap hook cannot see them.
#
# AND IT MUST RECORD THE REFUSALS. The gate is
#     abs((first + second*1000) - first) < 500   ==   abs(second*1000) < 500
# so every delay the snap hook sees ALREADY PASSED IT. A hook that recorded only the
# successful path would reproduce exactly the selection it was built to measure --
# "no large stage-1/stage-2 disagreements" would be true by construction. The except
# branch is therefore recorded too, with gate_passed=False, and it is the whole point
# of the column.
GATE_ORIG = "    def test_if_constant_good_delay(self):"
GATE_NEW = (
    "    # --- ci gate capture hook (lab, not committed) ---\n"
    "    def test_if_constant_good_delay(self):\n"
    "        import snap_capture as _sc\n"
    "        try:\n"
    "            _out = self._ci_orig_test_if_constant_good_delay()\n"
    "        except Exception as _e:\n"
    "            _sc.record_gate(self, getattr(self, '_ci_first', None),\n"
    "                            getattr(self, '_ci_second', None), False, repr(_e)[:120])\n"
    "            raise\n"
    "        _sc.record_gate(self, getattr(self, '_ci_first', None),\n"
    "                        getattr(self, '_ci_second', None), True, None)\n"
    "        return _out\n"
    "\n"
    "    def _ci_orig_test_if_constant_good_delay(self):")

# The two locals must be stashed on self before the gate can raise, so they survive
# into the except branch. Assigning them at their point of computation is the only
# place both are known.
STASH_ORIG = ("            delay_second_method = self.second_delay_test("
              "delay_first_method,ignore_audio_couple)")
STASH_NEW = (STASH_ORIG + "\n"
             "            self._ci_first = delay_first_method\n"
             "            self._ci_second = delay_second_method")


def apply(path):
    src = open(path).read()
    if MARK in src:
        return "ALREADY APPLIED -- refusing to double-apply"
    # EVERY anchor must be unique. A replace() on a pattern that occurs twice edits
    # the first and silently leaves the second, which is a half-instrumented file that
    # runs and lies. Checked for all three before any is applied.
    for name, pat in (("ORIG", ORIG), ("GATE_ORIG", GATE_ORIG), ("STASH_ORIG", STASH_ORIG)):
        if src.count(pat) != 1:
            return f"REFUSING: {name} expected exactly 1 occurrence, found {src.count(pat)}"
    out = src.replace(ORIG, NEW, 1)
    out = out.replace(STASH_ORIG, STASH_NEW, 1)   # stash BEFORE wrapping the method
    out = out.replace(GATE_ORIG, GATE_NEW, 1)
    fd, tmp = tempfile.mkstemp(suffix='.py'); os.close(fd)
    open(tmp, 'w').write(out)
    try:
        py_compile.compile(tmp, doraise=True)
    except py_compile.PyCompileError as e:
        os.unlink(tmp)
        return f"REFUSING: patched file does not compile -- {str(e)[:100]}"
    shutil.copy(path, path + '.pre-snap-hook')
    shutil.move(tmp, path)
    return "APPLIED; original kept at .pre-snap-hook"


if __name__ == '__main__':
    print("  " + apply(sys.argv[1]))
