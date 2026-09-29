#!/usr/bin/env python3
"""PreToolUse hook: search is for FINDING, reading is for UNDERSTANDING.

Operator rule (2026-09-29, after a session repeatedly read source through
`sed -n`, `grep -A`, `head` and `tail` slices, drew conclusions from them, and
carried a wrong claim -- "Cam96 has 12 axes" -- out of a one-line board grep
instead of reading the plan): **reading is mandatory; grep, sed, tail and head
are forbidden for reading.** lance-graph's `anti-pattern-matching.sh` only
REMINDED on most of this and records its own bypass (`cat x.rs | tail`); this
hook DENIES.

What is denied, and why each is reading rather than searching:

- Bash `sed` / `head` / `tail` / `awk` / `less` / `more` / `nl`, anywhere in
  the command, pipes included. A numeric slice has no semantic boundary, and
  "only limiting the display" is exactly the excuse that let slices of source
  through. `sed -i` is included: edits go through the Edit tool.
- Bash `cat` of a source/config file (not a redirect into one). Use Read.
- Bash `grep` / `rg` / `egrep` / `fgrep` / `ugrep` with a file or directory
  operand. Use the Grep tool, which reports what it caps.
- Any grep, in Bash or the Grep tool, asking for context lines
  (`-A` / `-B` / `-C` / `--context`, or the tool's `-A`/`-B`/`-C`/`context`):
  that is reading around a match.

What stays allowed:

- The Grep tool in `files_with_matches` or `count` mode, and `content` mode
  without context: locating candidates.
- Bash `grep` with NO file operand, reading a pipe from another command
  (`cargo test ... | grep "test result"`): filtering a process's own output is
  searching that output, not reading source.
- Read, always.

Tests: `.claude/hooks/tests/read-discipline.test.sh` (two-sided).
"""

import json
import re
import shlex
import sys

SLICERS = {"sed", "head", "tail", "awk", "gawk", "mawk", "less", "more", "nl"}
GREPS = {"grep", "rg", "egrep", "fgrep", "ugrep"}
CONTEXT_FLAG = re.compile(r"^-(?:[ABC]\d*|[A-Za-z]*[ABC]\d*|-context(?:=.*)?|-after-context(?:=.*)?|-before-context(?:=.*)?)$")
SOURCE_EXT = re.compile(
    r"\.(rs|toml|lock|md|py|c|cc|cpp|h|hpp|java|kt|ts|tsx|js|mjs|json|ya?ml|sql|"
    r"proto|sh|surql|ttl|html|csv|txt)$"
)

ZAP = "⚡ BLOCKED by read-discipline"


def deny(reason: str) -> None:
    print(
        json.dumps(
            {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": f"{ZAP}: {reason}",
                }
            }
        )
    )
    sys.exit(0)


HEREDOC = re.compile(r"<<-?[ \t]*(['\"]?)([A-Za-z_][A-Za-z0-9_]*)\1[^\n]*\n.*?\n[ \t]*\2[ \t]*(?=\n|$)", re.S)
SEPARATORS = {"|", "||", "&&", ";", "&", "\n", "(", ")", "$(", ";;", "|&"}
WRAPPERS = {"env", "sudo", "time", "nice", "command", "exec", "xargs", "then", "do", "else", "!"}


def segments(command: str):
    """Split a shell command into simple-command word lists, quote-aware.

    Heredoc bodies are dropped first (their text is data, not commands), then
    the rest is tokenized with quotes respected, so a commit message or a
    Python string that merely MENTIONS `sed` is not mistaken for running it.
    Pipes, `;`, `&&`, `||`, `&`, unquoted newlines and `$(...)` start a new
    command, so a slicer later in a pipeline or inside a substitution is seen.
    """
    text = HEREDOC.sub("<<HEREDOC", command.replace("\\\n", " "))
    text = text.replace("$(", " ( ").replace("`", " ( ")
    lexer = shlex.shlex(text, posix=True, punctuation_chars="();<>|&\n")
    lexer.whitespace = " \t\r"
    lexer.whitespace_split = True
    try:
        tokens = list(lexer)
    except ValueError:
        # Unbalanced quotes: fall back to a plain split, which can only err
        # towards seeing MORE command words, never fewer.
        tokens = re.split(r"[ \t\r]+", text)
    out, cur = [], []
    for tok in tokens:
        if tok in SEPARATORS or (tok and set(tok) <= set(";|&\n()")):
            if cur:
                out.append(cur)
            cur = []
        else:
            cur.append(tok)
    if cur:
        out.append(cur)
    cleaned = []
    for words in out:
        while words and (
            re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", words[0]) or words[0] in WRAPPERS
        ):
            words = words[1:]
            while words and words[0].startswith("-"):
                words = words[1:]
        if words:
            cleaned.append(words)
    return cleaned


def check_bash(command: str) -> None:
    for words in segments(command):
        prog = words[0].rsplit("/", 1)[-1]
        args = words[1:]
        if prog in SLICERS:
            deny(
                f"`{prog}` is forbidden. Reading is mandatory: open the file with the "
                "Read tool (whole semantic unit; continue from the exact next offset "
                "if it pages). To locate something, use the Grep tool."
            )
        if prog == "cat":
            # A redirect (`cat > f`, `cat >> f`, `cat <<EOF`) writes; only a
            # plain file operand reads. shlex splits `>`/`<` into own tokens.
            writes = any(a in {">", ">>", "<", "<<", "<<HEREDOC"} for a in args)
            operands = [a for a in args if not a.startswith("-") and a not in {">", ">>", "<", "<<"}]
            if not writes and any(SOURCE_EXT.search(a) for a in operands):
                deny("`cat` of a source/config file is reading through the shell. Use the Read tool.")
        if prog in GREPS:
            if any(CONTEXT_FLAG.match(a) for a in args):
                deny(
                    f"`{prog}` with context lines is reading around a match, not searching. "
                    "Find the file with the Grep tool, then Read it."
                )
            # Flags that take a value consume the next word.
            operands, skip = [], False
            for a in args:
                if skip:
                    skip = False
                    continue
                if a in {"-e", "-f", "-m", "--max-count", "-g", "--glob", "-t", "--type"}:
                    skip = True
                    continue
                if a.startswith("-"):
                    continue
                if a in {"<", ">", ">>", "2>&1"}:
                    break
                operands.append(a)
            has_e = any(a == "-e" or a.startswith("-e") for a in args)
            # The first operand is the pattern unless -e supplied it; anything
            # else is a file or directory the grep would read.
            file_operands = operands if has_e else operands[1:]
            if file_operands:
                deny(
                    f"`{prog}` over files in the shell is forbidden. Use the Grep tool "
                    "(files_with_matches / count) to locate, then Read."
                )


def check_grep_tool(tool_input: dict) -> None:
    for key in ("-A", "-B", "-C", "context"):
        if tool_input.get(key) not in (None, 0, "0"):
            deny(
                "Grep with context lines is reading around a match. Use "
                "files_with_matches or count to locate, then Read the file."
            )


def main() -> None:
    try:
        payload = json.load(sys.stdin)
    except json.JSONDecodeError:
        sys.exit(0)
    tool = payload.get("tool_name", "")
    tool_input = payload.get("tool_input") or {}
    if tool == "Bash":
        check_bash(tool_input.get("command", ""))
    elif tool == "Grep":
        check_grep_tool(tool_input)
    sys.exit(0)


if __name__ == "__main__":
    main()
