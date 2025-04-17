#!/bin/bash

cmd="python ./pitch_slope_and_other_features.py"

for dir in /blue/ejain/datasets/*/
do
    dir=${dir%*/}      # remove the trailing "/"
    echo "$dir"       # print the directory name
    $cmd --root_folder "$dir"
done
