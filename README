# Multi-Step Predictability

This project explores how the predictability of upcoming words varies across a text.

It uses GPT-2 to compute greedy predictions for multiple steps ahead and evaluates how well those predictions match the actual text, one word at a time.

The aim is to identify regions in a passage where future language is: (1) highly predictable multiple words ahead, (2) locally predictable for just the next word but not beyond, or (3) unpredictable. 

## Measuring Predictability

At each word in an input text:
- Uses **greedy decoding** with GPT-2 to predict `n` words into the future
- Compares each prediction with the actual word
- Records:
  - Whether the model’s prediction matched the true word (`match`)
  - The prediction’s **rank** (i.e., how likely the correct word was)
  - The model's **confidence score**
- Outputs both the predictions and their alignment with actual outcomes

## Project Structure

The project contains:
- A requirements.txt file including the dependencies needed
- A language_input folder in which to add .csv files containing language input. These files should be .csv files with at least one column called "word" in which each row is a word. You may need preprocessing to get to this point. There is an example file of artificial input in this folder.
- A scripts folder including a main.py file that performs the multistep_predictability evaluation. 
- A notebooks folder in which you can perform the multistep_predictablity evaluation interactively.
- A predictability_output folder in which you can store the evaluation data.