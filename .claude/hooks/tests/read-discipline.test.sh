#!/usr/bin/env bash
# Two-sided tests for .claude/hooks/read-discipline.py: every DENY case must be
# denied AND every ALLOW case must pass silently. A guard that fires on
# everything carries as little information as one that never fires.
#
# Run: bash .claude/hooks/tests/read-discipline.test.sh
set -u
HOOK="$(cd "$(dirname "$0")/.." && pwd)/read-discipline.py"
fail=0

run() { # $1 = JSON payload; prints the hook's decision: deny | allow
  out="$(printf '%s' "$1" | python3 "$HOOK")"
  if printf '%s' "$out" | jq -e '.hookSpecificOutput.permissionDecision == "deny"' >/dev/null 2>&1; then
    echo deny
  else
    echo allow
  fi
}

bash_json() { jq -n --arg c "$1" '{tool_name:"Bash", tool_input:{command:$c}}'; }

expect() { # $1 expected, $2 label, $3 payload
  got="$(run "$3")"
  if [ "$got" = "$1" ]; then
    echo "ok   $1  $2"
  else
    echo "FAIL expected $1, got $got: $2"
    fail=1
  fi
}

# ---- DENY: reading through the shell ----------------------------------------
expect deny "sed -n range of a source file" "$(bash_json "sed -n '120,180p' src/lib.rs")"
expect deny "head of a file"                 "$(bash_json 'head -50 crates/x/src/store.rs')"
expect deny "tail of command output"         "$(bash_json 'cargo test -p x 2>&1 | tail -30')"
expect deny "head on git output"             "$(bash_json 'git branch -r | head -15')"
expect deny "awk later in a pipeline"        "$(bash_json 'cd crates && ls | awk "{print \$1}"')"
expect deny "sed inside command substitution" "$(bash_json 'echo $(sed s/a/b/ notes.txt)')"
expect deny "sed -i edit"                    "$(bash_json "sed -i 's/a/b/' src/routes.rs")"
expect deny "grep over a file"               "$(bash_json 'grep -n "pub fn" src/auto_match.rs')"
expect deny "rg over a directory"            "$(bash_json 'rg -l deepnsm crates/')"
expect deny "grep -A context from a pipe"    "$(bash_json 'cargo test 2>&1 | grep -A5 FAILED')"
expect deny "grep -C2 on a file"             "$(bash_json 'grep -C2 foo a.md')"
expect deny "cat a source file"              "$(bash_json 'cat crates/x/src/lib.rs')"
expect deny "env-prefixed sed"               "$(bash_json 'LC_ALL=C sed -n 1,5p Cargo.toml')"
expect deny "sed after a newline"            "$(bash_json $'cd /tmp\nsed -n 1p x.rs')"
expect deny "Grep tool with context"         '{"tool_name":"Grep","tool_input":{"pattern":"x","-C":3}}'
expect deny "Grep tool with -A"              '{"tool_name":"Grep","tool_input":{"pattern":"x","-A":10,"output_mode":"content"}}'

# ---- ALLOW: searching, writing, and text that merely MENTIONS a slicer -------
expect allow "grep filtering cargo output"   "$(bash_json 'cargo test -p x 2>&1 | grep -E "FAILED|test result"')"
expect allow "commit message mentions sed/head" "$(bash_json 'git commit -q -m "stop using sed and head; tail is gone too"')"
expect allow "python heredoc mentions head"  "$(bash_json $'python3 - <<\'EOF\'\nprint("head -5 and sed are banned")\nEOF')"
expect allow "cat heredoc into a file"       "$(bash_json $'cat > notes.md <<\'EOF\'\nsed -n 1p x.rs\nEOF')"
expect allow "cat appending to a tag file"   "$(bash_json 'cat >> /tmp/tag.txt')"
expect allow "git log with a count"          "$(bash_json 'git log --oneline -3')"
expect allow "plain ls and wc"               "$(bash_json 'ls -la && wc -l x.md')"
expect allow "echo text containing awk"      "$(bash_json 'echo "awk is forbidden"')"
expect allow "Grep files_with_matches"       '{"tool_name":"Grep","tool_input":{"pattern":"x","output_mode":"files_with_matches"}}'
expect allow "Grep content without context"  '{"tool_name":"Grep","tool_input":{"pattern":"x","output_mode":"content"}}'
expect allow "Read tool"                     '{"tool_name":"Read","tool_input":{"file_path":"/x/src/lib.rs"}}'

if [ "$fail" -ne 0 ]; then
  echo "read-discipline: FAILURES"
  exit 1
fi
echo "read-discipline: all cases pass"
