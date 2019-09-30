#!/bin/bash

if [ "$#" -ne 3 ]; then
    echo $1 1.0 | python3 encoder.py
else
	echo $1 $2 | python3 encoder.py
fi