#!/usr/bin/env python3
"""Classify your own instruments by SAFEGUARD, not by form.

WHY THIS EXISTS. Three agents in one evening judged an instrument by its shape:
one found both its failures inside scripts; one rewrote a throwaway query as a
proper tool and the tool reproduced the defect it was built to fix; and I audited
my own numbers by "script versus ad-hoc" ONE HOUR after agreeing in writing that
form does not imply safeguards. None of the three was careless.

    THE PROPERTY THAT MATTERS IS NOT VISIBLE FROM THE OUTSIDE OF A FILE,
    SO THE EYE SUPPLIES THE ONE THAT IS.

This makes the invisible property visible. It is cheaper than a harness and needs
no coordination. It disagreed with my own judgement immediately: my throwaway
heredoc carried more of the discipline than two of my scripts did.

Two fields only, deliberately. A longer checklist is a checklist nobody runs.
"""
import re, sys, pathlib

CNM = re.compile(r'COULD NOT MEASURE|could_not_measure|not measurable|'
                 r'instrument-failed|no_reference|REFUSING|partial-coverage',
                 re.I)
DEN = re.compile(r'of \{len|/\{len|\{len\(|denominator|attempted|sampled \{|'
                 r'\bn=\{|print\(f?["\'].*\{[a-z_]+\}/\{[a-z_]+\}', re.I)
SWALLOW = re.compile(r'except\s+Exception[^\n]*:\s*\n\s*(continue|pass|return)')

def audit(p):
    s = pathlib.Path(p).read_text(errors='replace')
    return (bool(CNM.search(s)), bool(DEN.search(s)), len(SWALLOW.findall(s)))

if __name__ == "__main__":
    files = sys.argv[1:] or sorted(str(x) for x in pathlib.Path('.').glob('*.py'))
    print(f"{'instrument':<28}{'cannot-measure':>15}{'denominator':>13}{'silent-skip':>13}   verdict")
    for f in files:
        try: cnm, den, sw = audit(f)
        except Exception as e:
            print(f"{f:<28}  COULD NOT AUDIT: {str(e)[:40]}"); continue
        # the verdict is about what was PAID FOR, never about the file's form
        score = cnm + den
        verdict = ("lean hardest" if score == 2 and sw == 0 else
                   "usable, state the gap" if score >= 1 else
                   "LABEL BEFORE QUOTING")
        print(f"{f:<28}{'yes' if cnm else 'NO':>15}{'yes' if den else 'NO':>13}"
              f"{sw:>13}   {verdict}")
    print("\nA silent-skip branch that has never fired is indistinguishable from a")
    print("working one until it does. Measure whether it fired; do not assume either way.")
