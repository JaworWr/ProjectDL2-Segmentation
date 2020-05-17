#!/bin/bash

set -e

DATA_URL=http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
path=${1:-data}

ROOT=VOCdevkit/VOC2012
EXTRACT_DIRS="$ROOT/SegmentationClass $ROOT/JPEGImages $ROOT/ImageSets/Segmentation"

mkdir -p "$path"
echo "Extracting image data to $(readlink -f "$path")"
wget -nv --show-progress -O - $DATA_URL | tar -xC "$path" $EXTRACT_DIRS
