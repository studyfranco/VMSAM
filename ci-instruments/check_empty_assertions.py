#!/usr/bin/env python3
"""Assertions that cannot fail, and code that never runs. AST, over my own suites.

Built after two of mine printed green: `bool(accents) or True` in test 4e, and a whole
part-5 block that landed after `sys.exit(...)` while the suite reported `11 passed,
0 failed`. `vmsam-dev-2` built the same checker independently and its FIRST OUTPUT WAS
AN ACCUSATION AGAINST CORRECT CODE -- the paired raise idiom, where the predicate is
constant and THE MEASUREMENT IS IN THE CONTROL FLOW:

    try:
        f()                       # must raise
        check("it raises", False) # reached only if it did NOT
    except Error:
        check("it raises", True)  # reached only if it DID

So a constant inside try/except is exempt ONLY when both halves exist under the same
name. A LONE `check(name, True)` in an `except` tests nothing: if nothing raises it is
never reached and the suite is merely SHORTER, which only a count reveals.

WHAT THIS CANNOT SEE, stated so a clean run is not over-read: it reads SHAPES. A check
whose two sides are both zero AT RUNTIME -- dev-2's `len(x) == text.count(...)` passing
as `0 == 0` on a fixture with no change points -- is a property of the DATA and is
invisible here. ZERO EMPTY SHAPES IS NOT ZERO EMPTY CHECKS.

AND IT ONLY SEES FILES. I hit the `if True` class tonight in a script typed straight
into a shell, ran this over 73 committed files, and got 0 findings -- correctly, because
the mistake was not in anything committed. A CLEAN RUN HERE SAYS NOTHING ABOUT WORK THAT
NEVER BECAME A FILE, and the most careless things I write are the ones I do not save.
A SECOND BLIND SPOT, FOUND BY `vmsam-dev-2` ON THEIR OWN CODE, NOT BY THIS TOOL:

    30 checks before . 30 checks after . five written . none executed

They appended five checks after `sys.exit(...)` -- which this tool DOES catch -- and
then their correction deleted everything between the two epilogues, INCLUDING the block
itself. After that, there was no code after `sys.exit` any more. Nothing was stranded,
so there was nothing here to see, and this tool correctly reported nothing.

    A CHECKER FOR UNREACHABLE CODE CANNOT SEE CODE THAT HAS BEEN DELETED.

What saw it was the printed count, which had not moved. Every line printed was a genuine
pass and the exit code was 0 and correct about everything that ran. THE COUNT WAS THE
ONLY THING THAT SAID SO.

So this tool answers "is any assertion unable to fail"; it cannot answer "is every
assertion I wrote still here". That question has exactly one cheap instrument -- a
harness that prints `N check(s) held` rather than `PASS`, and a human who notices N did
not change. Keep the count in every harness. It is cruder than this file and it catches
what this file structurally cannot.
"""
import ast, sys, glob, collections

def is_const(node):
    if isinstance(node, ast.Constant):
        return True, repr(node.value)
    if isinstance(node, ast.BoolOp):
        for v in node.values:
            if isinstance(v, ast.Constant):
                if isinstance(node.op, ast.Or) and v.value:
                    return True, f"`or {v.value!r}` short-circuits the predicate"
                if isinstance(node.op, ast.And) and not v.value:
                    return True, f"`and {v.value!r}` short-circuits the predicate"
    if isinstance(node, ast.Compare) and len(node.ops) == 1 and isinstance(node.ops[0], ast.Eq):
        if ast.dump(node.left) == ast.dump(node.comparators[0]):
            return True, "X == X, true by construction"
    return False, ""

def scan(path):
    src = open(path, encoding='utf-8').read()
    tree = ast.parse(src)
    out = []
    # 1. constant predicates in check(...)-style calls, with the paired-idiom exemption
    paired = collections.Counter()
    for n in ast.walk(tree):
        if isinstance(n, ast.Try):
            names = collections.Counter()
            for sub in ast.walk(n):
                if isinstance(sub, ast.Call) and getattr(sub.func, 'id', None) == 'check' and sub.args:
                    if isinstance(sub.args[0], ast.Constant):
                        names[sub.args[0].value] += 1
            for k, v in names.items():
                if v >= 2:
                    paired[k] += 1
    # A CONSTANT `False` INSIDE AN `except` IS AN ERROR REPORT, NOT AN ASSERTION.
    # It fires exactly when something raised, so it cannot "fail to fail". My checker's
    # FIRST OUTPUT was an accusation against one of these -- the same way dev-2's first
    # output accused their paired raise idiom. Two independent checkers, two first runs,
    # both wrong about correct code, both by matching FORM.
    #
    # The dangerous twin is the opposite constant: a lone `check(name, True)` in an
    # `except` with no partner in the `try` is never reached when nothing raises, and
    # the suite is merely SHORTER. That one stays flagged.
    in_handler = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.ExceptHandler):
            for sub in ast.walk(n):
                if isinstance(sub, ast.Call):
                    in_handler.add(id(sub))
    for n in ast.walk(tree):
        if isinstance(n, ast.Call) and getattr(n.func, 'id', None) == 'check' and len(n.args) >= 2:
            name = n.args[0].value if isinstance(n.args[0], ast.Constant) else "<dynamic>"
            const, why = is_const(n.args[1])
            if not const or paired.get(name):
                continue
            arg = n.args[1]
            if id(n) in in_handler and isinstance(arg, ast.Constant) and arg.value is False:
                continue          # error report on the exception path
            out.append((n.lineno, f"check({name!r}, ...) predicate cannot fail: {why}"))
    # 2b. A CONDITIONAL EXPRESSION WITH A CONSTANT TEST -- `'yes' if True else 'no'`.
    # I wrote exactly that in an ad-hoc check tonight:
    #     print(f"foreign-language fill among them: {'NONE...' if True else ''}")
    # The loop above it did the work and happened to agree, but THE SENTENCE A READER
    # TAKES AWAY WAS NOT COMPUTED FROM ANYTHING. That is not an assertion that cannot
    # fail -- it is a CONCLUSION that cannot fail, which is worse, because an assertion
    # at least claims to be checking something.
    for n in ast.walk(tree):
        if isinstance(n, ast.IfExp):
            const, why = is_const(n.test)
            if const:
                out.append((n.lineno, f"conditional expression with a CONSTANT test: {why}"
                                      f" -- the branch is decided before the data"))

    # 2. `assert <constant>`
    for n in ast.walk(tree):
        if isinstance(n, ast.Assert):
            const, why = is_const(n.test)
            if const:
                out.append((n.lineno, f"assert with a constant test: {why}"))
    # 3. module-level code after sys.exit(...)  -- MY OWN FAILURE
    body = tree.body
    for i, n in enumerate(body):
        call = n.value if isinstance(n, ast.Expr) else None
        if isinstance(call, ast.Call) and ast.unparse(call.func) in ("sys.exit", "exit"):
            if i + 1 < len(body):
                out.append((body[i+1].lineno,
                            f"UNREACHABLE: {len(body)-i-1} top-level statement(s) after "
                            f"{ast.unparse(call.func)}() at line {n.lineno}"))
    return out

files = sorted(f for f in glob.glob('*.py') + glob.glob('lab/*.py')
               if f != 'check_empty_assertions.py')
total = 0
for f in files:
    try:
        hits = scan(f)
    except SyntaxError as e:
        print(f"  {f}: SYNTAX ERROR line {e.lineno} -- NOT SCANNED"); total += 1; continue
    for ln, msg in hits:
        print(f"  {f}:{ln}  {msg}"); total += 1
print(f"\n  {len(files)} file(s) scanned, {total} finding(s)")
print("  SHAPES ONLY. A check that is 0 == 0 because of its FIXTURE is invisible here.")
sys.exit(1 if total else 0)
