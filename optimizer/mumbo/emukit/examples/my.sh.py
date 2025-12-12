# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 11:19:11 2022

@author: fanny
"""


#!/bin/bash
#n_var=n_var, n_obj=n_obj, problem_name=problem, n_weight=100, seed=seed, n_const=0
#n_process=`cat /proc/cpuinfo |grep "processor"|wc -l`
n_process=29
PY=python3

seeds="1"
#seeds="11 12 13 14 15 16 17 18 19 20"

RUNNAME=python

pids=()
p=1
active_proc=0

wait_empty_processor(){
  echo "[SHELL] Wait empty processor $active_proc/$n_process"
  while [ $active_proc -ge $n_process ]
  do
    for j in $( seq 1 ${#pids[@]} )
    do
      #echo "check $j"
      if [ -z "${pids[$j]}" ]
      then
        echo "we have empty pids[${j}]"
      else
        if [ "${pids[$j]}" -ne -1 ]
        then
          if [ -z "`ps aux | awk '{print $2 }' | grep ${pids[$j]}`" ]
          then
            echo "[SHELL] $j:${pids[$j]} Finish $(date +"%T")"
            pids[$j]=-1
            let active_proc=$active_proc-1

          fi
        fi
      fi
    done
    sleep 120
  done
}

rm -rf ./$RUNNAME
ln -s `which $PY` ./$RUNNAME
for s in $seeds
      do

            cmd="./$RUNNAME BO_ES.py"
            echo $cmd
            $cmd &
            pids[$p]=$!
            echo "[SHELL]pids[${p}]=${pids[$p]} :$cmd Start $(date +"%T")"

            sleep 10
            wait_empty_processor
      done

echo "[SHELL] All runs added!"
