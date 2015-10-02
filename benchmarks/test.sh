#!/bin/bash

if [ ! -d "bin" ]; then
    echo "Could not find bin directory. Make sure you have compiled with 'make'"
else
    cd bin

    if [ -f "convolution" -a -f "convolution.cl" ]; then
        echo
        echo "Testing convolution..."
        echo
        ./convolution -t
    else
        echo "Could not find 'convolution' files. Make sure they compiled correctly"
    fi


    if [ -f "stereo" -a -f "stereo.cl" ]; then
        echo
        echo "Testing stereo..."
        echo
        ./stereo -t
    else
        echo "Could not find 'stereo' files. Make sure they compiled correctly"
    fi

    if [ -f "raycast" -a -f "raycast.cl" ]; then
        echo
        echo "Testing raycasting..."
        echo
        ./raycast -t
    else
        echo "Could not find 'raycasting' files. Make sure they compiled correctly"
    fi

    if [ -f "bilateral" -a -f "bilateral.cl" ]; then
        echo
        echo "Testing bilateral..."
        echo
        ./bilateral -t
    else
        echo "Could not find 'bilateral' files. Make sure they compiled correctly"
    fi

    cd ..
fi
