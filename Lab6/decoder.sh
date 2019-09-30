#!/bin/bash

if [ "$#" -ne 3 ]; then
    echo $1 $2 1.0 | python3 decoder.py
else
	echo $1 $2 $3 | python3 decoder.py
fi