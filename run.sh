#! /bin/sh

SEQ_NAME=$1
GENDER=$2
RESTART=$3

BLEND_PATH="./data/to_annotate/${SEQ_NAME}/annotate/${SEQ_NAME}.blend"
BLEND_pkl="./data/to_annotate/${SEQ_NAME}/annotate/smplx_param.pkl"
BLENDER_PATH="~/Desktop/blender/blender" #if linux
SYS="macos" # macos


function fresh_start() {
    python prepare_for_annotation.py --seq_name $SEQ_NAME --gender $GENDER
    if [ "$SYS" == "linux" ]; then
        $BLENDER_PATH $BLEND_PATH -P starter.py
    elif [ "$SYS" == "macos" ]; then
        /Applications/blender.app/Contents/MacOS/blender $BLEND_PATH -P starter.py
    fi
}

function resume(){
    BLEND_PATH="./data/to_annotate/${SEQ_NAME}/annotate/${SEQ_NAME}.blend"
    #linux
    if [ "$SYS" == "linux" ]; then
        $BLENDER_PATH $BLEND_PATH
    elif [ "$SYS" == "macos" ]; then
        /Applications/blender.app/Contents/MacOS/blender $BLEND_PATH
    fi
}

if [ "$RESTART" == "resume" ]; then
    if test -f "$BLEND_pkl"; then
        echo "Continuing from last saved"
        resume
    fi
elif [ "$RESTART" == "restart" ]; then
    echo "All saved progress will be removed"
    fresh_start
fi
