#! /bin/sh
SEQ_NAME=$1
GENDER=$2

BLEND_PATH="./data/to_annotate/${SEQ_NAME}/annotate/${SEQ_NAME}.blend"
BLEND_pkl="./data/to_annotate/${SEQ_NAME}/annotate/smplx_param.pkl"
BLENDER_PATH="~/Desktop/blender/blender"

if test -f "$BLEND_pkl"; then
        echo "Continuing from last saved"
        BLEND_PATH="./data/to_annotate/${SEQ_NAME}/annotate/${SEQ_NAME}.blend"
        #linux
        $BLENDER_PATH $BLEND_PATH

else
        python prepare_for_annotation.py --seq_name $SEQ_NAME --gender $GENDER
        $BLENDER_PATH $BLEND_PATH -P starter.py
