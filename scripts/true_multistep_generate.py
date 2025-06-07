# -*- coding: utf-8 -*-
"""

Define true_multistep_generate function which uses GPT2 to processes input text word-by-word, 
generates n_predict predicted words, and evaluate them against the actual n words. 
Evaluation metrics include the rank of the actual word in the model's predictions, the model's
confidence, and whether the predicted word matches the actual word.

"""

# Import dependencies
import torch
import numpy as np


# Define function
def true_multistep_generate(model, tokenizer, token_ids, context_len, df, n_predict=3, device="cpu"):

    # Get an empty tensor that has rows = number of words, columns = context length
    # Each row will correspond to a word in the transcript
    fill_value = tokenizer.pad_token_id or 0
    data = torch.full((len(token_ids), context_len), fill_value, dtype=torch.long)

    # This step fills in the context tokens for each row. 
    # These are the words up to and including
    # the current word, which will be used to pedict upcoming words.     
    # So, on the ith row, the context includes the words up to and including the ith word
    for i in range(len(token_ids)):
        context_tokens = token_ids[max(0, i - context_len + 1):i + 1]
        if len(context_tokens) > 0:
            data[i, -len(context_tokens):] = torch.tensor(context_tokens)

    # Initialize output containers
    pred_tokens_by_n = {f"pred_token_t+{n}": [np.nan] * len(token_ids) for n in range(1, n_predict + 1)}
    pred_words_by_n  = {f"pred_word_t+{n}":  [np.nan] * len(token_ids) for n in range(1, n_predict + 1)}
    match_by_n       = {f"match_t+{n}":      [np.nan] * len(token_ids) for n in range(1, n_predict + 1)}
    conf_by_n        = {f"conf_t+{n}":       [np.nan] * len(token_ids) for n in range(1, n_predict + 1)}
    rank_by_n        = {f"rank_t+{n}":       [np.nan] * len(token_ids) for n in range(1, n_predict + 1)}


    # Predict upcoming words and evaluate the predictions. 
    with torch.no_grad():
        # For each word...
        for i, seq in enumerate(data):
            # i = 4 # Used in debugging
            seq = data[i] # Get the context. This includes the word itself. E.g., if i = 5, seq = words 0-5 (first 6 words) 
            generated = seq.tolist() # To start off with, we copy the context to a "generated" list, to which we will iteratively append predicted (generated) words
            
            # Set up some lits to store predictions 
            preds, pred_words, confidences, logits_list = [], [], [], []
    
            # Now iteratively predict n upcoming words
            for step in range(n_predict):
                input_ids = torch.tensor([generated[-context_len:]], device=device) # Get the input used to generate predictions
                logits = model(input_ids).logits[0, -1].cpu() # Get logits for the next word given the input
                probs = torch.nn.functional.softmax(logits, dim=-1) # Get probs for the next word given the input
                top_token = torch.argmax(probs).item() # Get the top predicted word token
    
                entropy = -torch.sum(probs * torch.log(probs + 1e-12)).item() # Use the probs to calculate entropy for the next word
                confidence = 1 - (entropy / np.log(probs.size(0))) # Use entropy and probs to calculate confidence
    
                preds.append(top_token)  # Add the predicted token to preds
                pred_words.append(tokenizer.decode([top_token])) # Add the predicted word to pred_words
                confidences.append(confidence) # Append confidence
                logits_list.append(probs) # Append probs
    
                generated.append(top_token) # Do greedy prediction - append the token to generated, which will be iteratively used to generate more predictions
    
            # Align the predictions and their metrics to the ith word.
            # Add 2 more metrics - match and rank. 
            # Match is whether a word predicted for a position
            # actually matches the word that occurred in that position. 
            # Rank is the rank of the actual word within the predictions for the word in that position
            # When match = 1, rank = 1. When match = 0, rank can be >= 2. Ranks closer to 1 = better predictions.
            for n in range(1, n_predict + 1):
                # n = 1 # used in debugging
                target_index = i + n
                pred_token = preds[n - 1]
                pred_word = pred_words[n - 1]
                confidence = confidences[n - 1]
                probs = logits_list[n - 1]
    
                # Only compute match/rank if the target is in bounds
                if target_index < len(token_ids):
                    true_token = token_ids[target_index]
                    match = int(pred_token == true_token)
                    rank = torch.argsort(probs, descending=True).tolist().index(true_token) + 1
                else:
                    match = np.nan
                    rank = np.nan
    
                pred_tokens_by_n[f"pred_token_t+{n}"][i] = pred_token
                pred_words_by_n[f"pred_word_t+{n}"][i] = pred_word
                match_by_n[f"match_t+{n}"][i] = match
                conf_by_n[f"conf_t+{n}"][i] = confidence
                rank_by_n[f"rank_t+{n}"][i] = rank
    
    
    for key_dict in [pred_tokens_by_n, pred_words_by_n, match_by_n, conf_by_n, rank_by_n]:
        for key, values in key_dict.items():
            df[key] = values
            
    return df