#!/usr/bin/env python3
"""Names LOADED where nothing binds them. compileall cannot see this: a name bound
in no scope PARSES PERFECTLY. That defect cost the campaign container hours tonight.

FIRST VERSION FAILED ITS OWN POSITIVE CONTROL -- it reported ZERO on the file whose
defect we had already read, at the line we had already read, because it computed
module-level bindings by walking the WHOLE TREE. Every parameter of every function
became a module global, so every name was visible everywhere and nothing could ever
be reported. It would have swept the tree, said "no unbound names", and that sentence
would have been worth nothing while sounding like assurance.

So: SCOPES ARE REAL HERE. A scope's bindings stop at nested scope boundaries, and so
do its loads. Lambda arguments, comprehension targets and nested defs each get their
own scope, which is what a hand-rolled walker gets wrong and what produced eight
false positives out of nine in the peer sweep that found the original.

Remaining bias is toward FALSE NEGATIVES: class bodies are treated as visible to
nested functions (Python does not), and star-imported modules make a file
unanalysable and are REPORTED AS SUCH rather than silently passed.
"""
import ast, builtins, sys

BUILTINS = set(dir(builtins)) | {'__file__','__name__','__doc__','__spec__',
                                 '__package__','__loader__','__builtins__','__debug__'}
SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef,
          ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)

def _children(node):
    """Direct sub-AST of a scope, i.e. what belongs to THIS scope's body."""
    if isinstance(node, ast.Module): return list(node.body)
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)): return list(node.body)
    if isinstance(node, ast.Lambda): return [node.body]
    if isinstance(node, (ast.ListComp, ast.SetComp, ast.GeneratorExp)):
        return [node.elt] + list(node.generators)
    if isinstance(node, ast.DictComp): return [node.key, node.value] + list(node.generators)
    return []

def _own(node):
    """Every descendant belonging to this scope, NOT crossing into nested scopes."""
    for c in _children(node):
        stack = [c]
        while stack:
            n = stack.pop()
            yield n
            if isinstance(n, SCOPES):      # boundary: its insides are not ours
                continue
            stack.extend(ast.iter_child_nodes(n))

def _params(node):
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)): return set()
    a = node.args
    out = {x.arg for x in a.args + a.posonlyargs + a.kwonlyargs}
    if a.vararg: out.add(a.vararg.arg)
    if a.kwarg: out.add(a.kwarg.arg)
    return out

def bindings(node):
    out = _params(node)
    star = False
    for n in _own(node):
        if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)): out.add(n.id)
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)): out.add(n.name)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            for a in n.names:
                if a.name == '*': star = True
                else: out.add((a.asname or a.name).split('.')[0])
        elif isinstance(n, (ast.Global, ast.Nonlocal)): out.update(n.names)
        elif isinstance(n, ast.ExceptHandler) and n.name: out.add(n.name)
        elif isinstance(n, ast.MatchAs) and n.name: out.add(n.name)
        elif isinstance(n, ast.comprehension):
            for t in ast.walk(n.target):
                if isinstance(t, ast.Name): out.add(t.id)
    return out, star

def scopes(node, chain):
    yield node, chain
    for n in _own(node):
        if isinstance(n, SCOPES):
            yield from scopes(n, chain + [node])

def check(path):
    tree = ast.parse(open(path, encoding='utf-8', errors='replace').read())
    hits, star_any = [], False
    for sc, chain in scopes(tree, []):
        b, star = bindings(sc)
        star_any |= star
        vis = set(b)
        for p in chain:
            pb, ps = bindings(p); vis |= pb; star_any |= ps
        vis |= BUILTINS
        name = getattr(sc, 'name', type(sc).__name__)
        for n in _own(sc):
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and n.id not in vis:
                hits.append((path, n.lineno, name, n.id))
    return hits, star_any

if __name__ == '__main__':
    allh, unan = [], []
    for p in sys.argv[1:]:
        try: h, s = check(p)
        except SyntaxError as e:
            print(f"  {p}: DOES NOT PARSE at line {e.lineno} -- reported, not skipped"); continue
        if s:
            # A star-import makes the file unanalysable, so its hits are NOISE, not
            # findings. Emitting them anyway hands the reader a triage list -- which
            # is how a checker gets overridden and then protects nothing. 18 of the
            # first sweep's 21 hits were `Decimal` under `from decimal import *`.
            unan.append((p, len(h))); continue
        allh += h
    for p, ln, fn, nm in sorted(set(allh)):
        print(f"  {p}:{ln}  in {fn}()  ->  {nm}")
    # EXIT NON-ZERO ON A HIT. There was no `sys.exit` anywhere in this file: it found
    # the defect, printed it, AND RETURNED 0. Wired into a promotion gate as
    # `check && promote` it would have passed the very commit it was written to catch
    # -- an armed guard that cannot fire, proposed as the fix for a gate that could
    # not fire. A peer found it by RUNNING my negative control instead of accepting
    # "validated in both directions" from my message.
    #
    # Exit codes are part of the contract for anything that gates:
    #   0  no hits and every file analysable
    #   1  at least one unbound load
    #   2  nothing was analysable (all inputs star-imported or unparseable) -- NOT a
    #      pass, because silence from a tool that could not look is not a clean result
    print(f"  -- {len(set(allh))} unbound load(s) across {len(sys.argv)-1} file(s)"
          f"")
    for p, n in unan:
        print(f"  UNANALYSABLE (star-import): {p} -- {n} candidate(s) SUPPRESSED as unattributable")
    if allh:
        sys.exit(1)
    if unan and len(unan) == len(sys.argv) - 1:
        print("  NOTHING WAS ANALYSABLE -- this is not a pass")
        sys.exit(2)
    sys.exit(0)
