#!/bin/bash
if [ $1 == "train" ]; then
    python3 train.py $2 $3
elif [ $1 == "test" ]; then
    python3 test.py $2 $3
else
    echo "Invalid argument"
fi