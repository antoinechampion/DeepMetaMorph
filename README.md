# Generation of metamorphic code using Machine Learning

## Brief

The goal of this project is to provide a way to generate highly metamorphic assembly code using machine learning and recurrent neural networks. 

Results have shown potential but are highly state-of-the-art and shouldn't (can't) be used in production code.

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

### Step 1: Building a dataset

First, GDB need to parse programs to build a dataset containing sequences of x86 instructions with their associated effect on the state of a program.
Hence, you need to edit `gdb/food-for-extractor.txt` and provide paths to x86 executables compiled with debugging symbols. x86-64 programs, ELF executables, and any bytecode-compiled programs (Java, .NET...) are out of scope for this study.

Then, `cd` into the `gdb` folder and execute the following command:

    gdb-python27 -x deepmm-extract.py

The `deepmm-extract.py` script will process all the files you provided and store the result of its analysis into the `extracted` folder.

For information, the dataset that was used to train the weights in the `models/` folder was made using the analysis of the following programs:

- Audacity
- dvd+rw suite
- gdb
- GIMP
- mp3split
- OllyDbg
- SpaceSniffer

### Step 2: Tidying the data

At this step, the dataset is unstructured. We must use R to transform it into a consistent dataset.

Go into the `data_analysis/` folder and run `deepmm-data-analysis.R`.
This will output 3 files: `dimensions.csv.gz`, `input.csv.gz` and `output.csv.gz`. You need to extract them before training the network. Those can be large of several gigabytes.

### Step 3: Training the network

You need to run `training/deepmm-train.py`, which expect to find `dimensions.csv`, `input.csv` and `output.csv` in the `data_analysis` folder. By default, it will load the dataset chunk by chunk, use a batch size of 32 sequences, keep 10% of the data for validation and run 64 epochs.

It will output 3 model files into `models/` once trained.


