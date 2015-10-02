AUMA documentation
==================

This document provides the documentation for AUMA (AUto tuning by MAchine learning). It primarily focues on how to install and run AUMA and the included benchmarks. A higher level description of how AUMA works can be found in the papers:

* Falch T. L., Elster A. C. *Machine Learning Based Autotuning for OpenCL Performance Portability*, IPDPSWS 15
* Falch T. L., Elster A. C. *AUMA: a machine learning based autotuner*, JORS 

**Contents**

* [Installation](#installation)
* [Useage](#useage)
* [Benchmarks](#benchmarks)


Installation
============
<a name=installation></a>

Dependencies
------------

AUMA depends upon the following libraries:

* [FANN](http://leenissen.dk/fann/wp/) >= 2.1
* [pyfann](https://github.com/FutureLinkCorporation/fann2) >= 2.1

These are both abailable in the Ubuntu 14.04 repositories, and can be intalled with:

    sudo apt-get install libfann2 python-pyfann

for other operating systems, follow the links for installation instructions. Note that for pyfann, the pip/easy_install package contains a bug, and should instead be installed by downloading the source from GitHub followed by e.g.

	sudo python setup.py install
	
in the source directory.

OpenCL benchmarks require an OpenCL capable hardware device, as well as properly installed OpenCL software. Most recent high end GPUs from Nvidia and AMD, and recent high end CPUs from Intel and AMD are capable of running OpenCL. Installation instructions for the OpenCL software can be found at:

* [Intel](https://software.intel.com/en-us/articles/opencl-drivers)
* [AMD](http://developer.amd.com/tools-and-sdks/opencl-zone/amd-accelerated-parallel-processing-app-sdk/)
* [Nvidia](https://developer.nvidia.com/opencl)

Compilation
-----------

If OpenCL has been installed and configured correctly, the benchmarks can be compiled with:
	
	make

in the benchmark directory. If you wish to compile only the non-OpenCL benchmarks, use:

	make noocl
	
similarily,
	
	make ocl
	
will compile only the OpenCL benchmarks.
	
AUMA itself is a Python application, and does not require compilation

Validate Installation
---------------------

To quickly make sure that AUMA is instaled correctly, run:

	python AUMA.py ../benchmarks/dummy.txt
	
This will run the autotuner on a toy benchmark. If everything is working correctly, it should run withtout problems, and print out:

	Best configuration:
	[0, 0, 0, 0]

At the end.

To make sure that the OpenCL benchmarks are working, execute the test script in the benchmarks directory:

	chmod +x test.sh
	./test.sh
	
This will cause each benchmark to be executed, performing a self test.


Useage
======
<a name=useage></a>

To run AUMA, simply execute:

	python auma.py settings.txt
	
Where settings.txt is file with settings specified by the user. There are no command line options.

AUMA can only be used to auto-tune standalone executeables or scripts. If you only wish to auto-tune a part of an application, e.g. a single function or similar, you must add additional driver code to fulfill this requirement.

AUMA communicates with the application to be autotuned using plain text files, therefore the application to be autotuned must be able to read a file of tuning parameter values, and for each set of parameter values, configure the code to be auto-tuned as apropriate, execute and time it, and write the results to a new file.

The application to be auto tuned is executed twice, using a total of 4 files for communication.

The implementation and meaning of the auto-tuning parameters is left to the programmer of the code to be auto-tuned, from AUMAs point of view, they are simply variables which can take on different values, which affect performance. AUMA supports an arbitrary number of parameters, however, each of them can only take on consecutive integer values from 0 up to some limit.

Settings File
------------------

AUMA reads all its options from a settings file. The file consists of multiple lines, where each line has the format:

	SETTING:VALUE
	
The following is a list of the settings, and their possible values:

*	**COMMAND1** The command used to start the program to be autotuned for the first time it is executed.
	
	*Example value:* ./benchmark < infile1.txt > outfile1.txt
	
*	**COMMAND2** The command used to start the program to be autotuned for the second time it is executed.

	*Example value:* ./benchmark < infile2.txt > outfile2.txt
	
*	**FILE1** The file used for input the first tine the program to be auto tuned is executed.

	*Example value:* infile1.txt
	
*	**FILE2** The file used for output the first tine the program to be auto tuned is executed.

	*Example value:* outfile1.txt

*	**FILE3** The file used for input the second tine the program to be auto tuned is executed.

	*Example value:* infile2.txt
	
*	**FILE4** The file used for output the second tine the program to be auto tuned is executed.

	*Example value:* outfile2.txt
	
*	**PARAMETER_RANGES** A space separated list of integers, with one integer for each parameter, indicating the upper limit of possible values for the parameters. If the value is n, the parameter can take on values in the range [0 , n-1].

	*Example value:* 2 2 8 8 10
	
*	**N\_TRAINING\_SAMPLES** Number of parameter space points to be used for training the neural network.
	
*	**N\_SECOND\_STAGE** Number of parameter space points to be used for the second stage of the auto tuning process when a fixed size second stage is used. Cannot be used at the same times as **SECOND\_STAGE\_THRESHOLD**

	*Example value:* 100

*	**N\_SECOND\_STAGE\_MIN** Mininum number of valid samples to include in the second stage if threshold based second stage is used. Only valid if **SECOND\_STAGE\_THRESOLD** is specififed.

	*Example value:* 10

*	**N\_SECOND\_STAGE\_MAX** Maximum number of valid samples to include in the second stage if threshold based second stage is used. Only valid if **SECOND\_STAGE\_THRESOLD** is specififed.

*	**SECOND\_STAGE\_THRESHOLD** Probability threshold when theshold based second stage size is used. Cannot be used at the same time as **N_SECOND_STAGE**. Should be a number between 0 and 1.

	*Example value:* 0.2
	
*	**NETWORK\_SIZE** Number of neurons in the hidden layer of the neural network.

	*Example value:* 30
	
*	**KEEP_FILES** Whether of not to delete FILE1-FILE4 after completeion, 0 to delete, 1 to keep.

	*Example value:* 0

*	**K** The number of neural networks to use.

	*Example value:* 10
	
Communication files
-------------------

The format of the files used for communication between AUMA and the application to be autotuned are explained here. File 1 and 3 are written by AUMA, and the application to be autotuned must be able to read these files, file 2 and 4 are written by the application to we autotuned, and AUMA reads these files.

Invalid configurations should be indicated by using a negative value for the execution time.

**File 1**
The first line specifies the number of configurations, and the number of parameters for each configurations. The following lines have one configuration per line, with space separated parameters.

*Example:*

	4 3
	1 3 2
	2 1 3
	2 1 1
	1 3 1
	
**File 2**
One line for each configuration, with the corresponding execution time, seperated by spaces.

*Example:*

	1 3 2 3.21
	2 1 3 4.81
	2 1 1 3.19
	1.3.1 2.89

**File 3**
The first line specifies the number of configurations, and the number of parameters for each configuration. If threshold based second stage size is used, it also specifies the time threshold, the minimum number of valid second stage configurations and the maximum number of valid second stage parameters. The following lines have one configuraion per line, with space separated parameters.

*Example:*

	4 3 3.05 2 3
	1 3 2
	2 1 3
	2 1 1
	1 3 1

**File 4**
One line for each configuration, with the corresponding execution time, seperated by spaces (same as file 2).

*Example:*

	1 3 2 3.21
	2 1 3 4.81
	2 1 1 3.19
	1.3.1 2.89

Benchmarks
==========
<a name=benchmarks></a>

Name is distributed with the following OpenCL benchmarks:

* stereo
* convolution
* raycast

In addition, two much simpler benchmarks are included:

* test
* matmul

The simple benchmarks are included only to make it possible to easily verify that AUMA is working correctly. The OpenCL benchmarks are complex applications, which illustrates how code can be parameterized, and are used to demonstrate AUMAs ability to find good parameter settings.

OpenCL Benchmarks
-----------------

While performing different comutations, the OpenCL benchmarks all work in the same way, and take the same command line options. The benchmarks can work in one of three modes:

1. Generate random parameter space points, execute and time code for each of these points.
2. Read a file of paramenter space points, execute and time code for each of these points.
3. Self test mode. Runs once with default parameter values.

The default mode is mode 1\. When no command line arguments are specified, all possible parameter space points will be generated, and the code will be executed and run for all of them.

To ensure that no parameter configuration inadvertly causes the output to be incorrect, the output of the computations can be checked against the correct solution, stored in a file. Files with the correct solution are not distributed, but can be generated by the application in self test mode, using a parameter configuration known to work correctly.

The benchmarks has the following command line options:

	-h

Displays help message and exits.

	-n <number>
	
Number of configurations points to run and execute. Only valid in mode 1\. (When configuration points are randomly generated by the benchmark)

	-f <file>
	
Will read configuration points from <code><file></code> and execute and time code for all of them.

	-r
	
Use threshold based second stage size. Otherwise, fixed second stage size is assumed.
	
	-i <number>

Start at specified configuration number. If used in combination with -f, the first number of configurations points in the file will be skipped. Otherwise, the first number of configurations of the generated configurations will be skipped.
	
	-s
	
Seed random number generator with current timestamp. Otherwise, the same sequence of random numbers are used each time for generating the random configuration points.


	-m
	
Ignore invalid configurations (configurations for which the OpenCL code will not compile or run) when counting, can only be used toghether with -n


	-t
	
Self test mode. Will run the code once with a default configuration.


	-c <file>
	
File with correct output against which the output for each configuration point can be checked. Otherwise output will not be checked.


	-w <file>
	
Write output to a file (in binary format). Can only be used in combination with -t. The file can later be used as the correct output, in combination with -c.


	-l
	
List all available OpenCL devices and exit.


	-d <arg>

Select OpenCL device. Can be gpu/GPU or cpu/CPU in which case the first gpu or cpu found will be used. To select a specific device, use platformid,deviceid with the platform and device ids reported with -l.
