#!/usr/bin/env bash
#
# Creates a background image made from a sinusoid-noise pattern with
# a Perlin noise in bw over the top.
#
# The Perlin noise is generated with Fred Weinhaus' script perlin.sh:
#
# http://www.fmwconcepts.com/imagemagick/perlin/index.php
#
# Parameters:
#
# bg_perlin.sh $OUTFILE $SIZE $SCALE $BLEND

# OUTFILE - filename to write output to
# SIZE    - image size (for eg 224x224)
# SCALE   - scale of sinusoid noise
# BLEND   - balance between sinusoid noise and perlin noise

working="./working"
bgfile=$1
size=$2
scale=$3
blend=$4

mkdir -p $working

echo "SIZE = ${size}"

convert -size ${size}x${size} xc: +noise Random ${working}/random.png

convert ${working}/random.png  -channel R \
        -function Sinusoid 1,0 \
        -virtual-pixel tile -blur 0x${scale} -auto-level \
        -separate ${working}/red.jpg
convert ${working}/random.png  -channel G \
        -function Sinusoid 1,0 \
        -virtual-pixel tile -blur 0x${scale} -auto-level \
        -separate ${working}/green.jpg
convert ${working}/random.png  -channel B \
        -function Sinusoid 1,0 \
        -virtual-pixel tile -blur 0x${scale} -auto-level \
        -separate ${working}/blue.jpg

# convert -size ${size}x${size} xc: +noise Random ${working}/noise.png
# convert ${working}/noise.png -virtual-pixel tile -blur 0x12 -colorspace Gray -auto-level ${working}/noise.jpg

convert ${working}/red.jpg ${working}/green.jpg ${working}/blue.jpg -combine ${working}/base1.jpg


#convert -size ${size}x${size} gradient:black-white ${working}/fade.jpg
./perlin.sh ${size}x${size} ${working}/fade.jpg
composite -blend ${blend} ${working}/fade.jpg ${working}/base1.jpg $bgfile

