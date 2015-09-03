# Copyright (c) 2015, Thomas L. Falch
# For conditions of distribution and use, see the accompanying LICENSE and README files

# This file is part of the benchmarks for the AUMA machine learning based auto tuning application
# developed at the Norwegian University of Science and technology


all: stereo

stereo: stereo.c clutil.o configurations.o io.o parser.o
	gcc -std=c99 -Wall stereo.c configurations.o clutil.o io.o parser.o -lOpenCL -lm -o stereo

%.o : ../common/%.c
	gcc -std=c99 -Wall ../common/$*.c -c

clean:
	rm -f stereo *.o
