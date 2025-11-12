#!/bin/sh
# Shell script to run the sliding brick puzzle solver.
# This script mirrors the behaviour of the part one run.sh but
# supports the additional commands introduced in part two.  It
# forwards all arguments to the Python driver script.

if [ "$#" -eq 3 ]; then
  python3 sbp_Assignment_3.py "$1" "$2" "$3"
elif [ "$#" -eq 2 ]; then
  python3 sbp_Assignment_3.py "$1" "$2"
elif [ "$#" -eq 1 ]; then
  python3 sbp_Assignment_3.py "$1"
else
  echo "Usage: sh run.sh <command> [file] [move]" >&2
  exit 1
fi