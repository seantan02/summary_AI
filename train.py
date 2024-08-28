import argparse
import pandas as pd
from pathlib import Path
from time import sleep, time
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from eval import evaluate
from helper import get_vocabulary, tokenize_words, save_model, plot_loss,\
                    save_vocabulary, load_model, load_vocabulary, save_optimizer, load_optimizer
from NN.dataset import TextSimplificationDataset
from NN.seq2seq import Seq2Seq
from NN.transformer import Transformer
from shared import device, logger, SEED_NUMBER

torch.manual_seed(SEED_NUMBER)

# Training function
def train_epoch(train_loader, model, model_type, optimizer, criterion, padding_idx, teacher_forcing_p=0.5):
    model.train()

    total_loss = 0

    if model_type == "seq2seq":
        for src, trg in train_loader:
            optimizer.zero_grad()
            decoder_outputs, _, _ = model(src, trg, teacher_forcing_p=teacher_forcing_p)
            loss = criterion(decoder_outputs.transpose(1, 2), trg)
            
            loss.backward()
            optimizer.step()
            
            # Update total loss
            total_loss += loss.detach().item()

        return total_loss / len(train_loader)
    
    for src, trg in train_loader:
        optimizer.zero_grad()

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

        loss = criterion(output.transpose(1, 2), expected_output)
        
        loss.backward()
        optimizer.step()
        
        # Update total loss
        total_loss += loss.detach().item()
    
    return total_loss / len(train_loader)

def train(train_loader, val_loader, model, model_type, optimizer, n_epochs, model_save_path: Path, optimizer_save_path: Path,
            print_every=100, evaluate_every=10, padding_idx=0, start_epoch=0, resumed_loss=None, target_loss = None,
            min_lr=1e-6, model_backup_save_path: Path=None, optimizer_backup_save_path: Path=None, train_eval_loss_plot_path: Path=None,
            scheduler_step_size=10, scheduler_gamma=0.80):

    criterion = torch.nn.CrossEntropyLoss(ignore_index=padding_idx)  # Assuming 0 is the padding index
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)

    loss = 0  # Initialize loss

    try:
        best_loss = None
        best_val_loss = None

        if resumed_loss is not None:
            best_loss = resumed_loss
            loss = resumed_loss
            logger.info(f'Model resume training starting at epoch {start_epoch+1} with loss {resumed_loss}')

        start_time = time()

        training_losses = []
        validation_losses = []
        epoch_ranges = []

        # Define teacher_forcing_p by rules: 1 when loss > 5, 0.5 when loss > 2.5, 0.25 when loss > 1, 0 when loss <= 1 
        teacher_forcing_p = 1

        for epoch in range(start_epoch, n_epochs + 1):
            loss = train_epoch(train_loader=train_loader, 
                               model=model,  
                               model_type=model_type,
                               optimizer=optimizer, 
                               criterion=criterion,
                               padding_idx=padding_idx,
                               teacher_forcing_p=teacher_forcing_p)

            if model_type == "seq2seq":
                if loss > 5:
                    teacher_forcing_p = 9
                elif loss > 2.5 and loss <= 5 and teacher_forcing_p > 0.75:
                    teacher_forcing_p = 0.75
                elif loss > 1 and loss <= 2.5 and teacher_forcing_p > 0.5:
                    teacher_forcing_p = 0.5
                elif loss <= 1 and teacher_forcing_p > 0.20:
                    teacher_forcing_p = 0.20

                print(f"Teacher forcing p: {teacher_forcing_p}")

            if epoch % print_every == 0:
                logger.info('Epoch %d, progress: %d%%, loss: %.4f' % (epoch, epoch / n_epochs * 100, loss))
                print('Epoch %d, progress: %d%%, loss: %.4f' % (epoch, epoch / n_epochs * 100, loss))
                print(f'Learning rate: {scheduler.get_last_lr()[0]}')

            if evaluate_every and epoch % evaluate_every == 0:
                
                eval_loss, eval_accuracy = evaluate(model=model, 
                                                    model_type = model_type,
                                                    criterion=criterion,
                                                    val_loader=val_loader, 
                                                    padding_idx=padding_idx)
                
                logger.info('Epoch %d, progress: %d%%, training loss: %.4f, validation loss: %.4f, accuracy: %.4f' % 
                            (epoch, epoch / n_epochs * 100, loss, eval_loss, eval_accuracy))
                print('Epoch %d, progress: %d%%, training loss: %.4f, validation loss: %.4f, accuracy: %.4f' % 
                            (epoch, epoch / n_epochs * 100, loss, eval_loss, eval_accuracy))
                
                if best_val_loss is None or round(eval_loss, 3) < round(best_val_loss, 3):
                    best_val_loss = eval_loss
                    best_val_model_file_name = "val_best_"+model_save_path.parts[-1]
                    best_val_model_save_path = model_save_path.with_name(best_val_model_file_name)
                    save_model(model, best_val_model_save_path)

                    best_val_optimizer_name = "val_best_"+optimizer_save_path.parts[-1]
                    best_val_optimizer_save_path = optimizer_save_path.with_name(best_val_optimizer_name)
                    save_optimizer(optimizer, loss, epoch, best_val_optimizer_save_path)

                    logger.info(f'Model saved at epoch {epoch+1} with new best validation loss {eval_loss}')

                training_losses.append(loss) # Append the training loss
                validation_losses.append(eval_loss) # Append the validation loss
                epoch_ranges.append(epoch) # Append the epoch range

            if scheduler.get_last_lr()[0] > min_lr:
                scheduler.step()  # Step the scheduler only if the learning rate is greater than 1e-6

            if best_loss is None or round(loss, 3) < round(best_loss, 3):
                best_loss = loss

                save_model(model, model_save_path)
                save_optimizer(optimizer, loss, epoch, optimizer_save_path)

                if target_loss is not None and best_loss <= target_loss:
                    logger.info(f'Target loss reached at {best_loss}. Stopping training at epoch {epoch+1}')
                    break

            if time() - start_time >= 300:  # To give the machine a break
                logger.info("5 minute of running. Machine taking a 30 seconds break...")
                sleep(30)
                start_time = time()

    except KeyboardInterrupt:
        logger.error("Training interrupted by user")
    finally:
        if model_backup_save_path is not None:
            save_model(model, model_backup_save_path)

        if optimizer_backup_save_path is not None:
            save_optimizer(optimizer, loss, epoch, optimizer_backup_save_path)

        if train_eval_loss_plot_path is not None and len(epoch_ranges) > 0:
            plot_loss(training_losses, validation_losses, epoch_ranges, train_eval_loss_plot_path)

def main():
    # Create the parser
    parser = argparse.ArgumentParser(description="Arguments for training the model")

    # Add arguments
    parser.add_argument('version_to_load', type=int, help='Version of the model to load (Use -1 to start from scratch)')
    parser.add_argument('version_to_save', type=int, help='Version of the model to save as')
    parser.add_argument('training_file_path', type=str, help='Training file path (json format with "INPUT" and "TARGET" columns)')
    parser.add_argument('validation_file_path', type=str, help='Validation file path (json format with "INPUT" and "TARGET" columns)')
    parser.add_argument('batch_size', type=int, help='Batch size')
    parser.add_argument('model_type', type=int, help='Which model architecture are you using? (0 for Seq2Seq, 1 for Transformer)')
    parser.add_argument('--singularize_words', type=int, default=0, help='Should we singularize words? (0 for No, 1 for Yes)')
    
    #Transformer arguments
    parser.add_argument('--d_model', type=int, default=256, help='The number of expected features in encoder/decoder inputs')
    parser.add_argument('--n_head', type=int, default=4, help='The number of heads in the multiheadattention models')
    parser.add_argument('--max_input_seq_length', default=1000, type=int, help='The maximum input sequence length for the transformer')
    parser.add_argument('--max_output_seq_length', default=200, type=int, help='The maximum output sequence length for the transformer')
    parser.add_argument('--dim_feedforward', default=1024, type=int, help='The dimension of the feedforward network model')
    # Seq2Seq arguments
    parser.add_argument('--hid_dim', type=int, default=128, help='Hidden dimension for the Seq2Seq model')
    
    # Common arguments
    parser.add_argument('--num_layers', type=int, default=2, help='The number of sub-layers in both the encoder and decoder')
    parser.add_argument('--dropout_p', type=float, default=0.2, help='The dropout value for input and output layers')
    parser.add_argument('--hid_dropout_p', type=float, default=0.4, help='The dropout for hidden layers')
    parser.add_argument('--lr', type=float, default=0.0001, help='The learning rate')
    parser.add_argument('--scheduler_step_size', type=int, default=10, help='The step size for the scheduler')
    parser.add_argument('--scheduler_gamma', type=float, default=0.8, help='The gamma value for the scheduler')
    parser.add_argument('--min_lr',type=float, default=1e-5, required=False, help='The minimum learning rate')
    parser.add_argument('--n_epoch', type=int, default=100, help='The number of training epochs')
    parser.add_argument('--print_every', type=int, default=5, help='Interval for printing the training loss during the training')
    parser.add_argument('--evaluate_every', type=int, default=5, help='Interval for evaluating the model during training')
    parser.add_argument('--target_loss', type=float, default=0.5, help='The loss value to stop training')
    
    # Parse the arguments
    args = parser.parse_args()

    # Check arguments
    model_type = "seq2seq" if args.model_type == 0 else "transformer"

    if model_type == "transformer":
        if args.d_model % args.n_head != 0:
            raise Exception("d_model must be divisible by n_head")
    
    if model_type == "seq2seq":
        if args.hid_dim % 2 != 0:
            raise Exception("hid_dim must be divisible by 2")
        if args.num_layers < 2 and args.hid_dropout_p > 0:
            raise Exception("num_layers must be greater than 1 if hid_dropout_p is greater than 0")

    training_data_file_path = Path(args.training_file_path)
    train_df=pd.read_json(training_data_file_path)

    validation_data_file_path = Path(args.validation_file_path)
    val_df=pd.read_json(validation_data_file_path)

    vocab_dir_path = Path(__file__).parent / "checkpoints" / f"vocab_v{args.version_to_load}"
    inputW2t, inputT2w = load_vocabulary(vocab_dir_path, "input")
    outputW2t, outputT2w = load_vocabulary(vocab_dir_path, "output")

    singularize_word = True if args.singularize_words == 1 else False

    if inputW2t is None or outputW2t is None:
        # We will use whatever we have in training data to create the vocabulary + 10000 most common words
        with (Path(__file__).parent / "Data" / "vocab.txt").open(mode="r") as f:
            all_texts = f.readlines()
            common_words2t, common_t2w = get_vocabulary(all_texts, 
                                      lower_case=True, 
                                      convert_unicode_to_ascii=True,
                                      stop_words={"to", "the", "a", "an"},
                                      singularize_words=singularize_word,
                                      minimum_occurance=1)

        # Input vocabulary
        inputW2t, inputT2w = get_vocabulary(train_df["INPUT"].tolist(), 
                                  lower_case=True, 
                                  convert_unicode_to_ascii=True,
                                  stop_words={"to", "the", "a", "an"},
                                  singularize_words=singularize_word,
                                  minimum_occurance=5)  # Only include words that appear at least 2 times in the training data
        # Output vocabulary
        outputW2t, outputT2w = get_vocabulary(train_df["TARGET"].tolist(), 
                                  lower_case=True, 
                                  convert_unicode_to_ascii=True,
                                  stop_words={"to", "the", "a", "an"},
                                  singularize_words=singularize_word,
                                  minimum_occurance=5)  # Only include words that appear at least 2 times in the training data
        
        # Update the w2t and t2w to continue the index base on previous dictionary
        for key in common_words2t:
            if key in ["<PAD>", "<SOS>", "<EOS>", "<UNK>"]:
                continue

            if key not in inputW2t:
                inputW2t[key] = len(inputW2t)
                inputT2w[len(inputT2w)] = key

            if key not in outputW2t:
                outputW2t[key] = len(outputW2t)
                outputT2w[len(outputT2w)] = key
    
    vocab_save_path = Path(__file__).parent / "checkpoints" / f"vocab_v{args.version_to_save}"

    save_vocabulary(inputW2t, inputT2w, vocab_save_path, "input")
    save_vocabulary(outputW2t, outputT2w, vocab_save_path,  "output")

    if len(inputW2t) != len(inputT2w) or len(outputW2t) != len(outputT2w):
        logger.error(f"Vocabulary size mismatch")
        raise ValueError(f"Vocabulary size mismatch.")

    for key in inputT2w:
        if not isinstance(key, int):
            logger.error(f"Key {key} in t2w is not an integer")
            raise ValueError(f"Key {key} in t2w is not an integer")

    train_tokenized_input = [tokenize_words(sentence, inputW2t, lower_case=True, convert_unicode_to_ascii=True, stop_words={"to", "the", "a", "an"}, singularize_words=singularize_word) for sentence in train_df["INPUT"]]
    train_tokenized_output = [tokenize_words(sentence, outputW2t, lower_case=True, convert_unicode_to_ascii=True, stop_words={"to", "the", "a", "an"}, singularize_words=singularize_word) for sentence in train_df["TARGET"]]

    train_padded_input = pad_sequence(train_tokenized_input, batch_first=True, padding_value=0)
    train_padded_output = pad_sequence(train_tokenized_output, batch_first=True, padding_value=0)

    train_tensor_input = train_padded_input.clone().detach().to(device, dtype=torch.long)
    train_tensor_target = train_padded_output.clone().detach().to(device, dtype=torch.long)

    train_dataset = TextSimplificationDataset(train_tensor_input, train_tensor_target)

    # Validation dataset
    val_tokenized_input = [tokenize_words(sentence, inputW2t, lower_case=True, convert_unicode_to_ascii=True, stop_words={"to", "the", "a", "an"}, singularize_words=singularize_word) for sentence in val_df["INPUT"]]
    val_tokenized_output = [tokenize_words(sentence, outputW2t, lower_case=True, convert_unicode_to_ascii=True, stop_words={"to", "the", "a", "an"}, singularize_words=singularize_word) for sentence in val_df["TARGET"]]

    val_padded_input = pad_sequence(val_tokenized_input, batch_first=True, padding_value=0)
    val_padded_output = pad_sequence(val_tokenized_output, batch_first=True, padding_value=0)

    val_tensor_input = val_padded_input.clone().detach().to(device, dtype=torch.long)
    val_tensor_target = val_padded_output.clone().detach().to(device, dtype=torch.long)
    val_dataset = TextSimplificationDataset(val_tensor_input, val_tensor_target)

    BATCH_SIZE = args.batch_size

    # Create DataLoader for training and validation sets
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model_file_path = Path(__file__).parent / "checkpoints" / f"model_v{args.version_to_load}" / (model_type+".pth")
    if model_type == "seq2seq":
        hid_dim = args.hid_dim
        num_layers = args.num_layers

        model = load_model(Seq2Seq, model_file_path, device)

        if model is None:
            model = Seq2Seq(input_size=len(inputW2t), 
                            output_size=len(outputW2t),
                            hidden_size=hid_dim,
                            device=device,
                            num_layers=num_layers,
                            dropout_p=args.dropout_p,
                            hid_dropout_p=args.hid_dropout_p
                            )
    else:
        d_model = args.d_model
        n_head = args.n_head
        dim_feedforward = args.dim_feedforward

        model = load_model(Transformer, model_file_path, device)

        if model is None:
            model = Transformer(src_vocab_size=len(inputW2t),
                                tgt_vocab_size=len(outputW2t),
                                d_model=d_model,
                                nhead=n_head,
                                num_encoder_layers=args.num_layers,
                                num_decoder_layers=args.num_layers,
                                dim_feedforward=dim_feedforward,
                                dropout=args.dropout_p,
                                atten_dropout_p=args.hid_dropout_p,
                                max_input_seq_length=args.max_input_seq_length,
                                max_output_seq_length=args.max_output_seq_length
                                )

    model.to(device)

    start_epoch = 0
    loss = None
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    optimizer_file_path = Path(__file__).parent / "checkpoints" / f"model_v{args.version_to_load}" / "transformer_optimizer.pth"
    optimizer, start_epoch, _  = load_optimizer(optimizer, optimizer_file_path, device)

    N_EPOCH = args.n_epoch

    model_save_path = Path(__file__).parent / "checkpoints" / f"model_v{args.version_to_save}" / (model_type+".pth")
    optimizer_save_path = Path(__file__).parent / "checkpoints" / f"model_v{args.version_to_save}" / (model_type+"_optimizer.pth")
    model_backup_save_path = Path(__file__).parent / "models" / f"backup_model_v{args.version_to_save}" / (model_type+".pth")
    optimizer_backup_save_path = Path(__file__).parent / "models" / f"backup_model_v{args.version_to_save}" /(model_type+"_optimizer.pth")
    train_eval_loss_plot_path = Path(__file__).parent / "checkpoints" / f"model_v{args.version_to_save}" / (model_type+"_train_eval_loss_plot.jpg")

    train(train_loader=train_loader,
          val_loader=val_loader,
          model=model,
          model_type=model_type,
          optimizer=optimizer,
          n_epochs=N_EPOCH,
          model_save_path=model_save_path,
          optimizer_save_path=optimizer_save_path,
          print_every=args.print_every,
          evaluate_every=args.evaluate_every,
          padding_idx=outputW2t["<PAD>"],
          start_epoch=0,
          resumed_loss=loss,
          target_loss=args.target_loss,
          min_lr=args.min_lr,
          model_backup_save_path=model_backup_save_path,
          optimizer_backup_save_path=optimizer_backup_save_path,
          train_eval_loss_plot_path=train_eval_loss_plot_path,
          scheduler_step_size=args.scheduler_step_size,
          scheduler_gamma=args.scheduler_gamma)

if __name__ == "__main__":
    main()