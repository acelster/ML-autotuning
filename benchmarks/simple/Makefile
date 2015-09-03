# Copyright (c) 2015, Thomas L. Falch
# For conditions of distribution and use, see the accompanying LICENSE and README files

# This file is part of the benchmarks for the AUMA machine learning based auto tuning application
# developed at the Norwegian University of Science and technology


all : matmul test

matmul : matmul.c
	gcc -std=c99 matmul.c -o matmul -lm
	
test : test.c
	gcc -std=c99 test.c -o test -lm

clean :
	rm -f test matmul