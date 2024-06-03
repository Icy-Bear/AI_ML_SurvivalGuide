# !/bin/bash

if [ "$1" == "" ]; then
	echo "(-) Filename is not provied"
	exit 0
fi

mkdir 00_$1.md
mkdir 01_$1_Sklearn.py
mkdir 02_$1_FS.py

