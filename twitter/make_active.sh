#!/bin/bash

# 

TODAY=`date +%Y%m%d`



if [ -d "$TODAY" ]; then
    if [ -L "Active" ]; then
        ACTIVE=`readlink -f Active`
        unlink Active
        mv $ACTIVE Archive/
    fi
    ln -s $TODAY Active
fi
