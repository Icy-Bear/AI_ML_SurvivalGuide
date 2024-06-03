# !/bin/bash

if [ "$1" == "" ]; then
	echo "(-) Filename is not provied"
	exit 0
fi

mkdir $1
cd $1
touch 00_$1.md
touch 01_$1_Sklearn.py
touch 02_$1_FS.py

