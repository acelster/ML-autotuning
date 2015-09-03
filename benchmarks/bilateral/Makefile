# Copyright (c) 2015, Thomas L. Falch
# For conditions of distribution and use, see the accompanying LICENSE and README files

# This file is part of the benchmarks for the AUMA machine learning based auto tuning application
# developed at the Norwegian University of Science and technology


bilateral: bilateral.c io.o parser.o clutil.o configurations.o
	gcc -std=c99 -g -Wall bilateral.c io.o parser.o clutil.o configurations.o -lOpenCL -lm -o bilateral
	
%.o : ../common/%.c
	gcc -std=c99 -Wall ../common/$*.c -c
	
clean:
	rm -f bilateral *.o
