# Generation of metamorphic code using Machine Learning

## Brief

The goal of this project is to provide a way to generate highly metamorphic assembly code using machine learning and recurrent neural networks. 

The result has a lot of potential but is highly state-of-the-art and shouldn't (can't) be used in production code.

See TBA for more information.

## Dependencies
    - gdb-python27
    - R >= 3.2
        - purrr
        - magrittr
        - optional
        - dplyr
        - reshape2
        - ggplot2
        - jsonlite
        - binaryLogic
        - stringr
        - tidyr
    - python >= 3.5
        - numpy
        - pandas
        - tensorflow
        - keras > 2.0
    - CUDA > 9.1 and cuDNN are recommended for training the recurrent network


## How To

### Step 1: Creating a dataset

First, GDB need to parse programs to build a dataset containing sequences of x86 instructions and their associated effect on a program state.
Hence, you need to edit `gdb/food-for-extractor.txt` and provide paths to x86 executables compiled with debugging symbols. x86-64 programs, ELF executables, and any bytecode-compiled programs (Java, .NET...) are out of scope for this study.

Then, `cd` into the `gdb` folder and execute the following command:

    gdb-python27 -x deepmm-extract.py

The `deepmm-extract.py` script will process all the files you provided and store the result of its analysis into the `extracted` folder.

For information, the dataset that was used to train the weights in the model/ folder was made using the following programs:

- Audacity
- dvd+rw suite
- dvdauthor
- gdb
- GIMP
- HelloWorld in C
- mp3split
- mp3tag
- OllyDbg
- SpaceSniffer

### Step 2: Tidy the data

Data is now present but unstructured. We must use R to transform it into a consistent dataset.

Go into the `data_analysis` folder and run `deepmm-data-analysis.R`.
It will produce 3 output files: `dimensions.csv.gz`, `input.csv.gz` and `output.csv.gz`. You need to extract them before training the network, but they can be large of several gigabytes.

### Step 3: Training the network

You need to run `scripts/deepmm-train.py`, which expect to find `dimensions.csv`, `input.csv` and `output.csv` in the `data_analysis` folder. By default, it will load the dataset chunk by chunk, use a batch size of 32 sequences, 10% of the data for validation and do 64 epochs.

It will output 3 model files into the `models` folder once trained.


