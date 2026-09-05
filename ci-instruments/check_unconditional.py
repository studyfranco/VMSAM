#!/usr/bin/env python3
"""Is a module-level name bound on EVERY path, or only inside a conditional?

MY OWN unbound_names.py CANNOT SEE THIS CLASS. `IMAGE` is bound in module scope, so
scope analysis says it is bound; the fault is FLOW-SENSITIVE -- the binding sits
inside `if not done:` and two reads sit outside it. A checker that answers "is this
name bound somewhere" cannot answer "is it bound before it is read".
"""
import ast, sys

def _binds_on_every_path(node, name):
    """Does this statement bind `name` no matter which way control goes?

    A try/except that binds in the body AND in every handler binds unconditionally --
    my first version called that a conditional bind and flagged the fix as the defect.
    A checker that cannot tell a total binding from a partial one is a checker that
    reports its own coarseness as a finding.
    """
    def binds(stmts):
        return any(isinstance(x, ast.Name) and x.id == name and isinstance(x.ctx, ast.Store)
                   for s in stmts for x in ast.walk(s))
    if isinstance(node, ast.Assign):
        return any(isinstance(x, ast.Name) and x.id == name for x in node.targets)
    if isinstance(node, ast.Try):
        return binds(node.body) and node.handlers and all(binds(h.body) for h in node.handlers)
    if isinstance(node, ast.If):
        return binds(node.body) and node.orelse and binds(node.orelse)
    return False


def _top_level(tree):
    """Module body, with `if __name__ == "__main__":` unwrapped -- its body runs
    whenever the file is the program, which is the only way these runners are used."""
    out = []
    for n in tree.body:
        if (isinstance(n, ast.If) and isinstance(n.test, ast.Compare)
                and isinstance(n.test.left, ast.Name) and n.test.left.id == '__name__'):
            out.extend(n.body)
        else:
            out.append(n)
    return out


def conditional_binds(path, names):
    t = ast.parse(open(path).read())
    bad = []
    for name in names:
        top, nested = False, []
        for node in _top_level(t):               # main-guard body counts as top level
            if _binds_on_every_path(node, name):
                top = True
            for sub in ast.walk(node):
                if isinstance(sub, ast.Name) and sub.id == name and isinstance(sub.ctx, ast.Store):
                    if not (isinstance(node, ast.Assign) and sub in node.targets):
                        nested.append((sub.lineno, type(node).__name__))
        if not top and nested:
            bad.append((name, nested))
    return bad

if __name__ == '__main__':
    bad = conditional_binds(sys.argv[1], sys.argv[2].split(','))
    for n, where in bad:
        print(f"  {n}: bound ONLY inside {sorted({w for _, w in where})} at lines "
              f"{sorted({l for l, _ in where})} -- unbound on other paths")
    print(f"  conditionally-bound names: {len(bad)}")
    sys.exit(1 if bad else 0)
