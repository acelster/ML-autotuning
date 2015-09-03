# Copyright (c) 2015, Thomas L. Falch
# For conditions of distribution and use, see the accompanying LICENSE and README files

# This file is part of the benchmarks for the AUMA machine learning based auto tuning application
# developed at the Norwegian University of Science and technology


all: ocl noocl

ocl:
	mkdir -p bin
	$(MAKE) -C stereo
	mv stereo/stereo bin/
	cp stereo/stereo.cl bin/
	$(MAKE) -C raycasting
	mv raycasting/raycast bin/
	cp raycasting/raycast.cl bin/
	$(MAKE) -C convolution
	mv convolution/convolution bin/
	cp convolution/convolution.cl bin/
	$(MAKE) -C bilateral 
	mv bilateral/bilateral bin/
	cp bilateral/bilateral.cl bin/
	
noocl:
	mkdir -p bin
	$(MAKE) -C simple
	mv simple/matmul bin/matmul
	mv simple/test bin/test
	
clean:
	rm -rf bin
	$(MAKE) clean -C stereo
	$(MAKE) clean -C raycasting
	$(MAKE) clean -C convolution
	$(MAKE) clean -C bilateral 
	$(MAKE) clean -C simple
