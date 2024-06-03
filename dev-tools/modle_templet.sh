# !/bin/bash

if [ "$1" == "" ]; then
	echo "(-) Filename is not provied"
	exit 0
fi

mkdir $1.md
mkdir $1_Sklearn.py
mkdir $1_FS.py

