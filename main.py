import argparse
import torch
from torch.nn.utils.rnn import pad_sequence
from helper import load_model, load_vocabulary, tokenize_words
from NN.seq2seq import Seq2Seq
from NN.transformer import Transformer
from shared import device
import pandas as pd
from pathlib import Path
from search import get_stackoverflow_best_answer, get_stackoverflow_url
import json
import re

def main():
    # Create the parser
    parser = argparse.ArgumentParser(description="Arguments for using the model for summarization on error logs and searching it on google for stackoverflow.")

    # Add arguments
    parser.add_argument('model_type', type=int, help='Which model architecture are you using? (0 for Seq2Seq, 1 for Transformer)')
    parser.add_argument('error_file_path', type=str, help='Error log file path (absolute path)')
    parser.add_argument('output_file_path', type=str, help='The output file path (absolute path) without the extension')
    parser.add_argument('--limit', default=999, type=int, required=False, help='Limit the number of error logs to process (Optional)')

    # Parse the arguments
    args = parser.parse_args()

    # 1. Load the error logs
    error_log_path = Path(args.error_file_path)
    error_logs = pd.read_json(error_log_path)

    vocab_dir_path = Path(__file__).parent / "models"/ "best"
    inputW2t, inputT2w = load_vocabulary(vocab_dir_path, "input")
    outputW2t, outputT2w = load_vocabulary(vocab_dir_path, "output")

    # Model loading
    model_type = "seq2seq" if args.model_type == 0 else "transformer"

    model_file_path = Path(__file__).parent / "models"/ "best" / (model_type+".pth")
    if model_type == "seq2seq":
        model = load_model(Seq2Seq, model_file_path, device)
    else:
        model = load_model(Transformer, model_file_path, device)

    if model is None:
        raise Exception("Model not found. Please use train.py to create a model first.")

    # Move to device
    model = model.to(device)
    model.eval()

    error_info_headers = []
    error_info_without_header = []

    for error_info in error_logs["ERROR_INFO"]:
        error_text = str(error_info).split("\r")[0].strip()
        if "exception" in error_text.lower():
            error_info_parts = error_text.split(":")
            error_info_without_header.append(error_info_parts[1].strip())
            error_info_headers.append(error_info_parts[0].strip())
        else:
            error_info_without_header.append(error_text)
            error_info_headers.append("")

    tokenized_input = [tokenize_words(sentence, inputW2t, lower_case=True, convert_unicode_to_ascii=True, stop_words={"the", "a", "an"}) for sentence in error_info_without_header]

    padded_input = pad_sequence(tokenized_input, batch_first=True, padding_value=0)

    tensor_input = padded_input.clone().detach().to(device, dtype=torch.long)
    summarized_texts = []

    # Forward pass
    for to_decode in tensor_input:
        to_decode = to_decode.unsqueeze(0)
        if model_type == "seq2seq":
            decoded_tokens = model.predict(to_decode, outputW2t["<SOS>"], outputW2t["<EOS>"])
            decoded_sentence = " ".join([outputT2w[token] for token in decoded_tokens])
        else:
            decoded_tokens = model.predict(to_decode, outputW2t["<SOS>"], outputW2t["<EOS>"], device, model.max_output_seq_length)
            for decoded_token in decoded_tokens:
                decoded_sentence = " ".join([outputT2w[token.item()] for token in decoded_token])

        decoded_sentence = re.sub(r'<EOS>|<SOS>', '', decoded_sentence) # Remove special tokens
        decoded_sentence = re.sub(r'\s+', ' ', decoded_sentence)  # Remove extra whitesapce
        summarized_texts.append(decoded_sentence)

    error_logs_df = pd.DataFrame({"ERROR_INFO_HEADER": error_info_headers, "SUMMARIZED_ERROR_INFO": summarized_texts})
    error_logs_df["SUMMARY"] = error_logs_df["ERROR_INFO_HEADER"] + " " + error_logs_df["SUMMARIZED_ERROR_INFO"]
    
    # We search with the exception headers
    errors_with_header_by_occurance = error_logs_df['SUMMARY'].value_counts().head(args.limit)  # We sort the error by counting how many of the same error info occurs and take the top 3
    
    recommendations = dict()

    for error, count in errors_with_header_by_occurance.items():
        recommendations[error] = dict()
        splitted_error = error.split(" ")
        optimal_length_for_search = " ".join(splitted_error[:15])  # Limit it to 15 words of summary
        stack_overflow_url = get_stackoverflow_url(optimal_length_for_search, headless=True)
       
        if stack_overflow_url is None or len(stack_overflow_url) == 0:
            recommendations[error]["stack_overflow_url"] = "Not found"
            recommendations[error]["best_answer"] = "Not found"
            continue
       
        recommendations[error]["stack_overflow_url"] = stack_overflow_url

        best_answer = get_stackoverflow_best_answer(stack_overflow_url, headless=True)
        recommendations[error]["best_answer"] = best_answer

    # We search without the exception headers
    errors_no_header_by_occurance = error_logs_df['SUMMARIZED_ERROR_INFO'].value_counts().head(args.limit)  # We sort the error by counting how many of the same error info occurs and take the top 3

    for error, count in errors_no_header_by_occurance.items():
        if error in recommendations:
            continue

        recommendations[error] = dict()

        stack_overflow_url = get_stackoverflow_url(error, headless=True)
       
        if stack_overflow_url is None or len(stack_overflow_url) == 0:
            recommendations[error]["stack_overflow_url"] = "Not found"
            recommendations[error]["best_answer"] = "Not found"
            continue

    output_location = Path(args.output_file_path)

    with output_location.with_suffix('.json').open(mode='w') as json_file:
        json.dump(recommendations, json_file, indent=4)

    txt_content = ""

    for i, key in enumerate(recommendations):
        txt_content += "For error: " + str(key) + "\n"
        txt_content += "Stack Overflow URL: " + str(recommendations[key]["stack_overflow_url"]) + "\n"
        txt_content += "Best Answer: " + str(recommendations[key]["best_answer"]) + "\n\n"

    with output_location.with_suffix('.txt').open(mode='w') as txt_file:
        txt_file.write(txt_content)

if __name__ == "__main__":
    main()