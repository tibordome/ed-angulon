#!/bin/bash

cd ../output
bash create_dir.bash
cd ../src/angulon
rm class_ham.so; rm class_ham.cpp; rm utilities.so; rm utilities.cpp
python3 main.py
