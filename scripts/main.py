# -*- coding: utf-8 -*-
"""

This script evaluates how predictability varies over the course of a text.
This script:
- Loads and tokenizes a text
- Runs a function called true_multistep_generate which uses GPT2 to process the text
  word-by-word, and at each word, predict n words (or rather, tokens) in the future.
  The function calculates metrics that evaluate the prediction at each step.
  The function currently predicts multiple words into the future using a greedy
  search approach in which a next word is predicted, then the top prediction is 
  treated as an "observed" word that is appended to the context and used to predict
  the next word, and so on. 
- Runs a function to align each word with its predictability metrics
- Plot predictability

Note: 
- Since this script uses a large language model, it is best run using a GPU
- Your input should be in the form of a csv that contains at least one column
  called "word", where each row is a word. You may need some preprocessing to get to 
  this point. 
- For illustrative purposes, this script uses language input that has
  been artificially generated, which consists of several plausible sentences 
  concatenated together (e.g., The man walked down the street looking for a coffee shop)

IMPORTANT: 
- If running in Spyder, set your working directory to the project root folder: 
  'multistep_predictability'. 
- In Spyder, this is shown in the top right corner. This ensures relative paths work correctly.

"""

# --------------------- #   

# First, install and import dependencies 

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Determine script and project root directories
try:
    # If running as a script (__file__ is defined)
    script_dir = Path(__file__).resolve().parent
except NameError:
    # If running interactively (e.g., VS Code REPL)
    script_dir = Path().resolve() / "scripts"

# Ensure scripts directory is in sys.path for local imports
if str(script_dir) not in sys.path:
    sys.path.append(str(script_dir))

# Define the project root (parent of 'scripts' folder)
project_root = script_dir.parent

# Import custom modules from the scripts folder
from true_multistep_generate import true_multistep_generate
from multistep_predictability import multistep_predictability

pd.set_option('display.max_columns', 20)

# --------------------- #  

# Next, set the paths for the input and output files

# Identify path to language_input folder and language input file
# The language input file should be a .csv with at least one column
# called "word", and a row for each word.
# For simplicity, name your input file so that it first refers to
# the nature of the input, followed by _input. 
# Input path for language data
input_path = project_root / "language_input" / "artificial_test_input.csv"

# Extract base name from input file
base_name = input_path.stem.replace("_input", "")

# Output directory
output_dir = project_root / "predictability_output"
output_dir.mkdir(parents=True, exist_ok=True)

# File names for outputs
output_filename = f"{base_name}_multistep_greedy.tsv"
predictability_filename = f"{base_name}_multistep_greedy_predictability.tsv"

# Full paths for output files
output_path = output_dir / output_filename
predictability_path = output_dir / predictability_filename

# --------------------- #  

# Next, load text input

input_df = pd.read_csv(input_path)
assert "word" in input_df.columns, "input must contain a 'word' column"
input_df.head(10)

# --------------------- #  

# Now, set parameters for multistep prediction.

# The multistep prediction function will use GPT2 (Radford et al., 2019)
# The function will use the HuggingFace transformers library to work with GPT2

# Here, we will define arguments related to the model, including the model name,
# context length, number of words to predict, and device (ideally use cuda to utilize GPU)

# Model name - keep as gpt2
modelname = "gpt2"

# Context length - can increase or decrease 
context_len = 32

# Number of upcoming words to predict. Note that n_predict = 1 is the same as just
# predicting the next word. As n_predict increases, we will predict more words into
# the future. 
n_predict = 3
device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda", 0)
    print("Using cuda!")

# --------------------- # 

# Now tokenize the input 

# Tokenize input with spacing preserved
tokenizer = AutoTokenizer.from_pretrained(modelname)

# Add a space before each word to preserve spacing during tokenization
input_df.insert(0, "word_idx", input_df.index.values)
input_df["hftoken"] = input_df.word.apply(lambda x: tokenizer.tokenize(" " + x))

# Remove rows where the hftoken column is empty
input_df = input_df[input_df["hftoken"].map(lambda x: len(x) > 0)]

# Explode the hftoken column so that each token is in its own row
input_df = input_df.explode("hftoken", ignore_index=True)

# Add a column for token ids, which are the numerical representations of the tokens
input_df["token_id"] = input_df.hftoken.apply(tokenizer.convert_tokens_to_ids)


input_df.head()


# --------------------- # 

# Store the token ids

token_ids = input_df.token_id.tolist()

# --------------------- # 

# Run the multistep prediction function and store the output

model = AutoModelForCausalLM.from_pretrained(modelname).to(device).eval()

output_df = true_multistep_generate(model, tokenizer, token_ids, context_len, input_df.copy(), n_predict=n_predict, device=device)

# Save the file
output_df.to_csv(output_path, sep="\t", index=False)

# --------------------- # 

# Align the predictability metrics to corresponding words.

# Note that output_df is a dataframe where each row corresponds to a word in the input.
# For each word, we have made predictions for multiple upcoming words (number was
# specified in n_predict). We have also evaluated those predictions using some metrics,
# including whether it matched the actual word that occurred at that step, the confidence
# with which the word was predicted, and the rank of the actual word within the model's
# predictions (rank closer to 1 = better prediction)

output_df.head(15)


# Now we want to quantify: for each actual word in the input, how well was it
# predicted? And we want to quantify this for each step - e.g., how well was it predicted
# 1 word ago, 2 words ago, 3 words ago, etc.? 
# Our dataframe already has this info, but it's not aligned to each word in this way. 
# Currently, the rank, conf and match columns for each word refer to the predictions that 
# were made *based on that word*. But instead, we want to know how well that word was 
# *predicted by its preceding words*. 
# The multistep_predictability function transforms output to accomplish this.
# In the transformed output, each word will have n_predict rows (e.g., if we set
# n_predict = 3, then each word will have 3 rows.) 
# There is a "timestep" column that numbers these rows (e.g., if n_predict = 3, 
# the timestep column for a word will have values 1, 2, 3)
# Critically, there are two columns that denote the word's predictability for a given
# timestep: pred_rank and pred_conf. These predictability columns indicate how predictable
# a word was based on its preceding words. For example, for timestep = 1, the predctability
# columns indicate a word's predictability based on the context words up to and including
# the word 1 timestep ago (i.e., the immediately preceding word). For timestep = 2, 
# these columns indicate a word's predicability based on the context words up to the word
# 2 timesteps ago (i.e., the word before last), and so on. 


multistep_df = multistep_predictability(output_df, n_predict)

# Save the file
multistep_df.to_csv(predictability_path, sep="\t", index=False)

# Look:
multistep_df.head(15)

# --------------------- # 

# Evaluate predictability. 

# For each word, we can evaluate how predictable it was 1 word ago, 2 words ago, ... n_predict 
# words ago.
# To do this, we can examine the relationship between timestep (number of words ago) and 
# pred_rank (the rank of the actual word within the model's predictions, where ranks closer to 
# 1 indicate higher predictability)

# %%
sns.barplot(data=multistep_df, x="timestep", y="pred_rank", errorbar=("se", 1), capsize=0.2)
plt.show()
