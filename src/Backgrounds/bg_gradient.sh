#!/usr/bin/env bash
#
# Creates a background image made from a radial gradient between two 
# colours with Perlin noise over the top.
#
# The Perlin noise is generated with Fred Weinhaus' script perlin.sh
#
# http://www.fmwconcepts.com/imagemagick/perlin/index.php
#
# Parameters:
#
# bg_gradient $OUTFILE $SIZE $MODE $ATT $BLEND $LEVELS $BLUR
#
# OUTFILE - filename to write output to
# SIZE    - image size (for eg 224x224)
# MODE    - color|gray parameter for noise
# ATT     - attenuation parameter for perlin.sh
# BLEND   - balance of gradient/noise
#           0 = gradient, 100 = noise
# LEVELS  - level balance like "5%,95%" - everything < 5% pushed to black,
#           everything above 95% pushed to white
# BLUR    - blur RxS (radius x sigma)

file=$1
geom="$2x$2"
mode=$3
att=$4
working="./working"
blend=$5
if [ -e ./last_colour.txt ]
then
    go="$(cat ./last_colour.txt)"
else
    go="$(rnd_gradient.py)"

fi
gn="$(./rnd_gradient.py)"
gradient="radial-gradient:${go}-${gn}"
echo $gn > ./last_colour.txt
echo "gradient: $gradient"
echo "go = $go"

level=$6
blur=$7

echo "geom ${geom}"
echo "perlin mode ${mode}"
echo "attenuation ${att}"
echo "blend ${blend}"
echo "gradient ${gradient}"
echo "levels ${level}"
echo "blur ${blur}"

/Users/mike/Desktop/DeepDream/deepdream/scripts/perlin.sh ${geom} -m ${mode} -a ${att} ${working}/perlin.jpg

convert -size ${geom} ${gradient} ${working}/fade.jpg 
composite -blend ${blend} ${working}/perlin.jpg ${working}/fade.jpg ${working}/comp.jpg
convert -level ${level} -blur ${blur} ${working}/comp.jpg ${working}/bg1.jpg
convert ${working}/bg1.jpg -colorspace rgb -type truecolor ${file}


echo ${file}

