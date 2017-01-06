# Copyright (c) 2015, Thomas L. Falch
# For conditions of distribution and use, see the accompanying LICENSE and README files

# This file is part of the benchmarks for the AUMA machine learning based auto tuning application
# developed at the Norwegian University of Science and technology

median: median.c clutil.o configurations.o parser.o io.o
	gcc -std=c99 -Wall -O3 -D IMAGE_WIDTH=3072 -D IMAGE_HEIGHT=3072 -D FILTER_WIDTH=5 -D FILTER_HEIGHT=5 -D PADDING=2 median.c clutil.o configurations.o parser.o io.o -lOpenCL -lm -o median
	
median_alt: median.c clutil.o configurations.o parser.o io.o
	gcc -std=c99 -Wall -O3 -D IMAGE_WIDTH=4608 -D IMAGE_HEIGHT=4608 -D FILTER_WIDTH=3 -D FILTER_HEIGHT=3 -D PADDING=1 median.c clutil.o configurations.o parser.o io.o -lOpenCL -lm -o median_alt
	
	
%.o : ../common/%.c
	gcc -std=c99 -Wall -O3 ../common/$*.c -c
	
clean:
	rm -f median *.o median_*
