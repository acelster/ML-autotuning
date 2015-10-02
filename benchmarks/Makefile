# Copyright (c) 2015, Thomas L. Falch
# For conditions of distribution and use, see the accompanying LICENSE and README files

# This file is part of the benchmarks for the AUMA machine learning based auto tuning application
# developed at the Norwegian University of Science and technology


all: ocl noocl

ocl: bin/stereo bin/raycast bin/convolution bin/bilateral

bin/stereo :
	mkdir -p bin
	$(MAKE) -C stereo
	mv stereo/stereo bin/
	cp stereo/stereo.cl bin/

bin/raycast :
	mkdir -p bin
	$(MAKE) -C raycasting
	mv raycasting/raycast bin/
	cp raycasting/raycast.cl bin/

bin/convolution :
	mkdir -p bin
	$(MAKE) -C convolution
	mv convolution/convolution bin/
	cp convolution/convolution.cl bin/

bin/bilateral:
	mkdir -p bin
	$(MAKE) -C bilateral 
	mv bilateral/bilateral bin/
	cp bilateral/bilateral.cl bin/
	
noocl: bin/test bin/matmul

bin/test:
	mkdir -p bin
	$(MAKE) -C simple
	mv simple/test bin/test

bin/matmul:
	mkdir -p bin
	$(MAKE) -C simple
	mv simple/matmul bin/matmul
	
clean:
	rm -rf bin
	$(MAKE) clean -C stereo
	$(MAKE) clean -C raycasting
	$(MAKE) clean -C convolution
	$(MAKE) clean -C bilateral 
	$(MAKE) clean -C simple
