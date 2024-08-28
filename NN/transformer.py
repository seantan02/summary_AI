import torch
import math

class PositionalEncoding(torch.nn.Module):
    def __init__(self, seq_length, d_model, dropout=0.1):
        """
        Initializes the PositionalEncoding module.
        
        Args:
        seq_length (int): Maximum sequence length.
        d_model (int): Dimension of the model.
        dropout_p (float): Dropout probability (default: 0.1).
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        # Create a long enough position encoding
        pe = torch.zeros(1, seq_length, d_model)
        position = torch.arange(0, seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Adds positional encoding to the input embedding.
        
        Args:
        x (Tensor): Input embedding tensor of shape (batch_size, seq_len, d_model).
        
        Returns:
        Tensor: Output tensor with positional encoding added.
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerEncoderLayer(torch.nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout_p=0.1, atten_dropout_p=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = torch.nn.MultiheadAttention(d_model, nhead, dropout=atten_dropout_p, batch_first=True)
        self.linear1 = torch.nn.Linear(d_model, dim_feedforward)
        self.linear2 = torch.nn.Linear(dim_feedforward, d_model)

        self.norm = torch.nn.LayerNorm(d_model)

        self.dropout = torch.nn.Dropout(dropout_p)

        self.activation = torch.nn.ReLU()

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        if src_mask is not None and src_key_padding_mask is not None:
            if src_mask.dtype != src_key_padding_mask.dtype:
                raise Exception("src_mask and src_key_padding_mask must have the same dtype")
        # Norm before using
        src = self.norm(src)
        
        # Self-attention
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        # Add and norm 1
        add_and_norm = self.norm(src + src2)
        add_and_norm = self.dropout(add_and_norm)

        # Feedforward
        squared_relu = torch.square(self.activation(self.linear1(add_and_norm)))
        feed_forward = self.linear2(squared_relu)

        # Add and norm 2
        add_and_norm2 = self.norm(add_and_norm + feed_forward)
        add_and_norm2 = self.dropout(add_and_norm2)

        return add_and_norm2

class TransformerEncoder(torch.nn.Module):
    def __init__(self, input_size, d_model, nhead, num_layers, dim_feedforward=2048, dropout=0.1, atten_dropout_p=0.2, max_seq_length=1000):
        super(TransformerEncoder, self).__init__()
        
        self.d_model = d_model
        self.embedding = torch.nn.Embedding(input_size, d_model)
        self.positional_encoding = PositionalEncoding(max_seq_length, d_model, dropout=dropout)
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, atten_dropout_p)
        self.layers = torch.nn.ModuleList([encoder_layer for _ in range(num_layers)])

    def forward(self, src, mask=None, src_key_padding_mask=None):
        output = self.embedding(src) * math.sqrt(self.d_model)
        output = self.positional_encoding(output)
        for layer in self.layers:
            output = layer(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
        return output


class TransformerDecoderLayer(torch.nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, attn_dropout_p=0.1):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = torch.nn.MultiheadAttention(d_model, nhead, dropout=attn_dropout_p, batch_first=True)
        self.multihead_attn = torch.nn.MultiheadAttention(d_model, nhead, dropout=attn_dropout_p, batch_first=True)
        
        self.linear1 = torch.nn.Linear(d_model, dim_feedforward)
        self.linear2 = torch.nn.Linear(dim_feedforward, d_model)
        
        self.norm = torch.nn.LayerNorm(d_model)

        self.dropout = torch.nn.Dropout(dropout)
        
        self.activation = torch.nn.ReLU()

        self.output_linear = torch.nn.Linear(d_model, d_model)
        
    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):

        if tgt_mask is not None and tgt_key_padding_mask is not None:
            if tgt_mask.dtype != tgt_key_padding_mask.dtype:
                raise Exception(f"tgt_mask and tgt_key_padding_mask must have the same dtype: {tgt_mask.dtype} != {tgt_key_padding_mask.dtype}")
            
        if memory_mask is not None and memory_key_padding_mask is not None:
            if memory_mask.dtype != memory_key_padding_mask.dtype:
                raise Exception(f"memory_mask and memory_key_padding_mask must have the same dtype: {memory_mask.dtype} != {memory_key_padding_mask.dtype}")
        """
        This takes in input that should have been embedded + positional encoded.
        After this function, there should be a linear out to transform output into desired shape.

        1. Normalize src
        2. Dropout1 on normalized src + masked self-attention
        3. Normalize from step 2
        4. Output from 3. as query, memory as key and value for multihead attention
        5. Dropout 2 on (output from 4. + add_norm_1)
        6. Norm from step 5
        7. Using output from 6.: Linear1 -> ReLU -> Square -> Linear2
        8. Dropout 3 on output from 7. and output from step 6., then perform add and norm
        9. Return output from step 8.
        """
        # Step 1
        normalized_tgt = self.norm(tgt)

        masked_multihead_atten = self.self_attn(normalized_tgt, normalized_tgt, normalized_tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        
        # Step 2
        output = self.dropout(normalized_tgt+masked_multihead_atten)

        # Step 3
        add_norm_1 = self.norm(output)

        # Step 4
        multihead_atten = self.multihead_attn(query=add_norm_1, key=memory, value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        
        # Step 5
        output = self.dropout(add_norm_1+multihead_atten)

        # Step 6
        add_norm_2 = self.norm(output)

        # Step 7
        # Feedforward
        squared_relu = torch.square(self.activation(self.linear1(add_norm_2)))
        feed_forward = self.linear2(squared_relu)

        # Step 8
        output = self.dropout(add_norm_2+feed_forward)
        return self.norm(output)

class TransformerDecoder(torch.nn.Module):
    def __init__(self, output_size, d_model, nhead, num_layers, dim_feedforward=2048, dropout=0.1, atten_dropout_p=0.2, max_seq_length=5000):
        super(TransformerDecoder, self).__init__()
        
        self.d_model = d_model
        self.embedding = torch.nn.Embedding(output_size, d_model)
        self.positional_encoding = PositionalEncoding(max_seq_length, d_model, dropout=dropout)
        
        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, atten_dropout_p)
        self.layers = torch.nn.ModuleList([decoder_layer for _ in range(num_layers)])
        
        self.output_linear = torch.nn.Linear(d_model, output_size)
        
    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        
        output = self.embedding(tgt) * math.sqrt(self.d_model)
        output =self.positional_encoding(output)

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                        tgt_key_padding_mask=tgt_key_padding_mask,
                        memory_key_padding_mask=memory_key_padding_mask)
        
        output = self.output_linear(output)

        return output
    

class Transformer(torch.nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers, 
                 dim_feedforward=2048, dropout=0.1, atten_dropout_p=0.2, max_input_seq_length=1000, max_output_seq_length=1000):
        super(Transformer, self).__init__()
        self.kwargs = {'src_vocab_size': src_vocab_size, 'tgt_vocab_size': tgt_vocab_size, 'd_model': d_model, 
                       'nhead': nhead, 'num_encoder_layers': num_encoder_layers, 'num_decoder_layers': num_decoder_layers, 
                       'dim_feedforward': dim_feedforward, 'dropout': dropout, 'atten_dropout_p': atten_dropout_p, 
                       'max_input_seq_length': max_input_seq_length, 'max_output_seq_length': max_output_seq_length
                       }
        
        self.max_input_seq_length = max_input_seq_length
        self.max_output_seq_length = max_output_seq_length
        self.encoder = TransformerEncoder(src_vocab_size, d_model, nhead, num_encoder_layers, dim_feedforward, dropout, atten_dropout_p, max_input_seq_length)
        self.decoder = TransformerDecoder(tgt_vocab_size, d_model, nhead, num_decoder_layers, dim_feedforward, dropout, atten_dropout_p, max_output_seq_length)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None, src_key_padding_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        memory = self.encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        output = self.decoder(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                              tgt_key_padding_mask=tgt_key_padding_mask,
                              memory_key_padding_mask=memory_key_padding_mask)
        return output

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def predict(self, src, start_symbol, end_symbol, device, max_output=100):
        self.eval()  # Set the model to evaluation mode
        
        src = src.to(device)
        batch_size = src.shape[0]
        
        # Encode the source sequence
        memory = self.encoder(src)
        
        # Initialize the target sequence with the start symbol
        prediction = torch.ones(batch_size, 1).fill_(start_symbol).type(torch.long).to(device)
        
        for i in range(max_output - 1):
            # Decode
            out = self.decoder(prediction, memory)
            # Select the token with the highest probability (greedy decoding)
            _, next_word = out[:, -1:, :].topk(1, dim=-1)
            next_word = next_word.squeeze(0)
            # Concatenate the predicted word to the output sequence
            prediction = torch.cat([prediction, next_word], dim=1)
            # Check if all sequences have predicted the end symbol
            if (next_word == end_symbol).all():
                break
        
        return prediction