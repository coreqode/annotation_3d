#! /bin/sh
SEQ_NAME='abhi'
GENDER='male'
BLEND_PATH="./data/to_annotate/${SEQ_NAME}/annotate/${SEQ_NAME}.blend"

python prepare_for_annotation.py --seq_name $SEQ_NAME --gender $GENDER
/Applications/blender.app/Contents/MacOS/blender $BLEND_PATH -P starter.py

# /Applications/Blender.app/Contents/Resources/2.93/python/bin/python3.9 -m pip install joblib
