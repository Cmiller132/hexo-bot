"""Best-effort JS syntax check for the edited app.js (no node available).
Tries esprima/pyjsparser; falls back to a string/template/comment-aware
bracket-balance scan so an unbalanced edit is caught."""
from pathlib import Path

# The static app.js in THIS checkout (repo-relative), so the check tracks the
# app.js actually shipped here rather than an external worktree.
APP = (
    Path(__file__).resolve().parents[2]
    / "packages" / "hexo_frontend" / "python" / "hexo_frontend" / "static" / "app.js"
)


def _bracket_balance_scan(src: str) -> None:
    """String/template/comment/regex-aware bracket balance. Raises AssertionError
    on the first mismatch or an unclosed bracket."""

    i, n = 0, len(src)
    stack = []
    pairs = {")": "(", "]": "[", "}": "{"}
    line = 1
    prev_sig = ""  # last significant (non-space) char, to disambiguate / as regex
    while i < n:
        c = src[i]
        if c == "\n":
            line += 1; i += 1; continue
        if c == " " or c == "\t" or c == "\r":
            i += 1; continue
        # line comment
        if c == "/" and i + 1 < n and src[i+1] == "/":
            while i < n and src[i] != "\n":
                i += 1
            continue
        # block comment
        if c == "/" and i + 1 < n and src[i+1] == "*":
            i += 2
            while i + 1 < n and not (src[i] == "*" and src[i+1] == "/"):
                if src[i] == "\n":
                    line += 1
                i += 1
            i += 2; continue
        # regex literal: a '/' is a regex when the previous significant token is
        # not a value (i.e. an operator, '(', ',', '=', 'return', etc.). Skip the
        # whole /.../ including [char classes] so its brackets/quotes don't corrupt
        # the scan.
        if c == "/" and prev_sig in "(,=:[!&|?{;" + "":
            j = i + 1
            in_class = False
            ok = False
            while j < n:
                cj = src[j]
                if cj == "\\":
                    j += 2; continue
                if cj == "\n":
                    break
                if cj == "[":
                    in_class = True
                elif cj == "]":
                    in_class = False
                elif cj == "/" and not in_class:
                    ok = True
                    j += 1
                    break
                j += 1
            if ok:
                # skip regex flags
                while j < n and src[j].isalpha():
                    j += 1
                i = j
                prev_sig = "/"
                continue
        # strings
        if c in "\"'":
            q = c; i += 1
            while i < n and src[i] != q:
                if src[i] == "\\":
                    i += 2; continue
                if src[i] == "\n":
                    line += 1
                i += 1
            i += 1; prev_sig = "x"; continue
        # template literal (handle ${ ... } nesting shallowly)
        if c == "`":
            i += 1
            while i < n and src[i] != "`":
                if src[i] == "\\":
                    i += 2; continue
                if src[i] == "$" and i + 1 < n and src[i+1] == "{":
                    depth = 1; i += 2
                    while i < n and depth:
                        if src[i] == "{":
                            depth += 1
                        elif src[i] == "}":
                            depth -= 1
                        elif src[i] == "\n":
                            line += 1
                        i += 1
                    continue
                if src[i] == "\n":
                    line += 1
                i += 1
            i += 1; prev_sig = "x"; continue
        if c in "([{":
            stack.append((c, line))
        elif c in ")]}":
            assert stack and stack[-1][0] == pairs[c], (
                f"[jscheck] BRACKET MISMATCH at line {line}: '{c}' vs open "
                f"{stack[-1] if stack else ('<none>', '?')}"
            )
            stack.pop()
        prev_sig = c
        i += 1
    assert not stack, f"[jscheck] UNCLOSED bracket from line {stack[-1][1]}: '{stack[-1][0]}'"


def test_app_js_is_syntactically_balanced() -> None:
    src = APP.read_text(encoding="utf-8")
    for mod in ("esprima", "pyjsparser"):
        try:
            m = __import__(mod)
        except ImportError:
            continue
        if mod == "esprima":
            m.parseScript(src)  # raises on a parse error
        else:
            m.parse(src)
        return  # a real JS parser accepted it
    # No JS parser available: fall back to the bracket-balance scan.
    _bracket_balance_scan(src)
