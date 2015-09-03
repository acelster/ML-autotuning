# Copyright (c) 2015, Thomas L. Falch
# For conditions of distribution and use, see the accompanying LICENSE and README files

# This file is part of the benchmarks for the AUMA machine learning based auto tuning application
# developed at the Norwegian University of Science and technology


convolution: convolution.c clutil.o configurations.o parser.o io.o
	gcc -std=c99 -Wall convolution.c clutil.o configurations.o parser.o io.o -lOpenCL -lm -o convolution 
	
%.o : ../common/%.c
	gcc -std=c99 -Wall ../common/$*.c -c
	
clean:
	rm -f convolution *.o
