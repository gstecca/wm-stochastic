#!/bin/bash
# 4 parallel workers x 2 threads on the 8-core VM
cd ~/qmaxtest
grep -v '^$' jobs.txt | xargs -P 4 -L 1 sh -c 'venv/bin/python run_qmax_test.py "$0" "$1" 2 results/res_$$.csv >> results/stdout.txt 2>&1 || echo "FAIL $0 $1" >> results/fail.txt'
echo DONE > results/DONE
