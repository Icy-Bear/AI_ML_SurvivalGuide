# !/bin/bash

if [ "$1" == "" ]; then
	echo "(-) Filename is not provied"
	exit 0
fi

mkdir $1
cd $1
touch 1_$1.md
touch 2_$1_Sklearn.py
touch 3_$1_FS.py

