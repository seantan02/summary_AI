# Error Log Solution Finder

This project implements a transformer-based neural network for text summarization, with a focus on summarizing error logs and performing related Google searches, and finding potential code fixes on stackoverflow.

## Table of Contents
- [Overview](#overview)
- [Files](#files)
- [Usage](#usage)
- [Contributing](#contributing)

## Overview

This neural network project utilizes a transformer architecture to perform text summarization. It is designed to summarize error logs without the header such as "System.HTTPException...:", and used the summarized error log as google searching key.

Then the program will fetch the first stackoverflow url found and look for the potential best solution.

## Files

The project consists of three main Python files that could be used via command line:

1. `train.py`: Handles the training of the model, either from scratch or from a saved version.
2. `eval.py`: Evaluates a saved model version.
3. `main.py`: Uses the trained model to summarize error logs and perform Google searches based on the summaries.

## Usage

### Training the Model

To train the model:

- run the train.py and specify the arguments needed in command line. Read the section in main.py where arguments are specified.
    - python train.py --argument1 xxxx ....

Notes:
1. Model should be trained and engineered based on the goal usage and dataset size.
2. Any slight changes could yield a better model, therefore it is beneficial to try out different model's settings and choose the "best" you have.
3. Current best setting found with a small datasets (around 500 pairs) are :
    Transformer:
    - features_dim = 512
    - batch_size = cube root of dataset pair sizes round to power of 2
    - number of heads = 8
    - layers = 3
    - dropout = 0.2, attn_dropout = 0.5
    - feedforward_dim = 2048

    Seq2Seq:
    - hidden dimension of 256
    - 2 layers
    - dropout 0.2, hid_dropout 0.5


### Evaluating the Model

To evaluate a saved model:

- run the eval.py and specify the arguments needed in command line. Read the section in main.py where arguments are specified.
    - python eval.py --argument1 xxxx ....

### Running the Main Program

To summarize error logs and perform Google searches:

- run the main.py and specify the arguments needed in command line. Read the section in main.py where arguments are specified.
    - python main.py --argument1 xxxx ....

## License

- This program is an asset to Sean Tan Siong Ann. Please do not use it for commercial purpose unless permission given by author.