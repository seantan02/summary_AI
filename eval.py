import argparse
import torch
from NN.seq2seq import Seq2Seq
from NN.transformer import Transformer
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from NN.dataset import TextSimplificationDataset
from helper import tokenize_words, load_model, load_vocabulary
from shared import device, SEED_NUMBER, logger
from pathlib import Path
import pandas as pd

torch.manual_seed(SEED_NUMBER)

def evaluate(model, model_type, criterion, val_loader, padding_idx):
    model.eval()

    with torch.no_grad():
        total_accuracy = 0
        total_loss = 0

        if model_type == "seq2seq":
            for src, trg in val_loader:
                decoder_outputs, _, _ = model(src, trg, teacher_forcing_p=0)
                loss = criterion(decoder_outputs.transpose(1, 2), trg)
                
                comparison_result = (decoder_outputs.argmax(dim=-1) == trg).to(dtype=torch.long)
                total_correct = comparison_result.sum()

                loss = criterion(decoder_outputs.transpose(1, 2), trg)
                
                # Update total loss
                total_loss += loss.detach().item()

                # Update total accuracy
                total_accuracy += total_correct.item()/(comparison_result.shape[0] * comparison_result.shape[1])

            return total_loss / len(val_loader), total_accuracy / len(val_loader)
        
        for src, trg in val_loader:
            src_key_padding_mask_bool = (src == padding_idx).to(device, dtype=torch.bool)
            src_key_padding_mask = torch.zeros_like(src_key_padding_mask_bool, dtype=torch.float32)
            src_key_padding_mask[src_key_padding_mask_bool] = float('-inf')
            fed_trg = trg[:, :-1]

            tgt_key_padding_mask_bool = (fed_trg == padding_idx).to(device, dtype=torch.bool)
            tgt_key_padding_mask = torch.zeros_like(tgt_key_padding_mask_bool, dtype=torch.float32)
            tgt_key_padding_mask[tgt_key_padding_mask_bool] = float('-inf')
            tgt_mask = model.generate_square_subsequent_mask(fed_trg.size(1)).to(device)
            output = model(src, fed_trg, src_mask=None, tgt_mask=tgt_mask, memory_mask=None, src_key_padding_mask=src_key_padding_mask, tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=None)

            expected_output = trg[:, 1:]

            comparison_result = (output.argmax(dim=-1) == expected_output).to(dtype=torch.long)
            total_correct = comparison_result.sum()

            loss = criterion(output.transpose(1, 2), expected_output)
            
            # Update total loss
            total_loss += loss.detach().item()

            # Update total accuracy
            total_accuracy += total_correct.item()/(comparison_result.shape[0] * comparison_result.shape[1])

        return total_loss / len(val_loader), total_accuracy / len(val_loader)

def main():
    # Create the parser
    parser = argparse.ArgumentParser(description="Arguments for evaluating the model")

    # Add arguments
    parser.add_argument('version_to_load', type=str, help='Version of the model to load (Use -1 to start from scratch)')
    parser.add_argument('validation_file_path', type=str, help='Training file path (json format with "INPUT" and "TARGET" columns)')
    parser.add_argument('model_type', type=int, help='Which model architecture are you using? (0 for Seq2Seq, 1 for Transformer)')
    parser.add_argument('--batch_size', default=4, type=int, required=False, help='Batch size (default: 4)')
    parser.add_argument('--test_sentence', type=str, required=False, help='A sentence to test the model (Optional)')
    # Parse the arguments
    args = parser.parse_args()

    vocab_dir_path = Path(__file__).parent / "checkpoints" / f"vocab_v{args.version_to_load}"
    inputW2t, inputT2w = load_vocabulary(vocab_dir_path, "input")
    targetW2t, targetT2w = load_vocabulary(vocab_dir_path, "target")

    file_path = Path(args.validation_file_path)
    validation_df=pd.read_json(file_path)

    # Use the model that generalized the best
    model_type = "seq2seq" if args.model_type == 0 else "transformer"

    model_file_path = Path(__file__).parent / "checkpoints" / f"model_v{args.version_to_load}" / ("val_best_"+model_type+".pth")
    if model_type == "seq2seq":
        model = load_model(Seq2Seq, model_file_path, device)
    else:
        model = load_model(Transformer, model_file_path, device)

    if model is None:
        raise Exception("Model not found. Please use train.py to create a model first.")

    # Move to device
    model = model.to(device)

    tokenized_input = [tokenize_words(sentence, inputW2t, lower_case=True, convert_unicode_to_ascii=True) for sentence in validation_df["INPUT"]]
    tokenized_output = [tokenize_words(sentence, targetW2t, lower_case=True, convert_unicode_to_ascii=True) for sentence in validation_df["TARGET"]]

    padded_input = pad_sequence(tokenized_input, batch_first=True, padding_value=0)
    padded_output = pad_sequence(tokenized_output, batch_first=True, padding_value=0)

    tensor_input = padded_input.clone().detach().to(device, dtype=torch.long)
    tensor_target = padded_output.clone().detach().to(device, dtype=torch.long)

    BATCH_SIZE = args.batch_size

    # Calculate the sizes for training and validation sets
    val_dataset = TextSimplificationDataset(tensor_input, tensor_target)

    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    if args.test_sentence:
        logger.info(f"Test sentence: {args.test_sentence}")
        print(f"Test sentence: {args.test_sentence}")
        
        test_input = tokenize_words(args.test_sentence, inputW2t, lower_case=True, convert_unicode_to_ascii=True).to(device)
        print(f"Tokenized test sentence: {test_input}")
        test_input = test_input.unsqueeze(0)
        # Prediction
        if model_type == "seq2seq":
            decoded_tokens = model.predict(test_input, targetW2t["<SOS>"], targetW2t["<EOS>"])
            logger.info(f'Output sentence: {" ".join([targetT2w[token] for token in decoded_tokens])}')
            print(f'Output sentence: {" ".join([targetT2w[token] for token in decoded_tokens])}')
        else:
            decoded_tokens = model.predict(test_input, targetW2t["<SOS>"], targetW2t["<EOS>"], device, model.max_output_seq_length)
            for decoded_token in decoded_tokens:
                logger.info(f'Output sentence: {" ".join([targetT2w[token.item()] for token in decoded_token])}')
                print(f'Output sentence: {" ".join([targetT2w[token.item()] for token in decoded_token])}')

    loss, accuracy = evaluate(model, model_type, torch.nn.CrossEntropyLoss(ignore_index=targetW2t["<PAD>"]), val_loader, padding_idx=targetW2t["<PAD>"])
    logger.info(f"Evaluation loss: {loss}")
    logger.info(f"Evaluation accuracy: {accuracy*100:.2f}%")

    print(f"Evaluation loss: {loss}")
    print(f"Evaluation accuracy: {accuracy*100:.2f}%")

if __name__ == "__main__":
    main()

