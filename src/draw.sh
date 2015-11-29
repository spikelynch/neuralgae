#!/usr/bin/env bash

# draw.sh $OUTDIR $BASEFILE $CONFIG $SIZE $SCALE $BLEND

working="./working"
outdir=$1
output=$2
config=$3
size=$4
scale=$5
blend=$6

mkdir -p $working
mkdir -p $outdir

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


convert -size ${size}x${size} gradient:black-white ${working}/fade.jpg
composite -blend ${blend} ${working}/fade.jpg ${working}/base1.jpg ${working}/base.jpg

#convert ${working}/graybase.jpg -colorspace rgb -type truecolor ${working}/base.jpg

./dream.py --config $config --basefile ${output} ${working}/base.jpg ${outdir}
