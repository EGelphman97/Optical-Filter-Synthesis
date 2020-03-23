# Optical-Filter-Synthesis
This repository has code for a Python program that designs and  synthesizes an MA/FIR digital optical filter. There are four main source code files:
    1. synthesisDriver.py
    2. synthesisDriverV2.py
    3. latticeFilterSynthesis.py
    4. designFilter.py
    
The files synthesisDriver.py and synthesisDriverV2.py are the files that actually run the program, and import designFilter.py and latticeFilterSynthesis.py as dependencies. synthesisDriver.py is designed to be run in a UNIX terminal and synthesisDriverV2 is designed to run in any Python IDE. The program requires an input file specifying the filter design parameters and an output file that stores the lengths of the various components needed for layout. To run synthesisDriver.py in the UNIX terminal, use the following commands

$ python3 synthesisDriver.py -l inputFilename.txt outputFilename.txt //if using list input format
$ python3 synthesisDriver.py -t inputFileWithTableName.txt outputFilename.txt //if using table format

The file latticeFilterSynthesis.py contains the implementation of the synthesis algorithm and its "inverse", as outlined in Section 4.5 of Madsen and Zhao

The file designFilter.py contains the implementation of all functions needed for filter design and the conversion between the continuous-time and discrete-time frequency domains. 

The file ARFilterSynthesis.py is still under development. It is an incomplete implementation of an extension of the synthesis algorithm for MA/FIR filter to the AR/all pole case. This is outlined in Section 5.2 of Madsen and Zhao
    
