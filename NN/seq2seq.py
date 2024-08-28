import torch
import random

from shared import device

# Encoder class
class EncoderRNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, dropout_p=0.2, hid_dropout_p = 0.5):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.kwargs = {"input_size": input_size, "hidden_size": hidden_size, "num_layers": num_layers, "dropout_p": dropout_p}

        self.embedding = torch.nn.Embedding(input_size, hidden_size)
        self.gru = torch.nn.GRU(hidden_size, hidden_size, num_layers=num_layers, dropout=hid_dropout_p, batch_first=True)
        self.dropout = torch.nn.Dropout(dropout_p)

    def forward(self, input):
        embedded = self.dropout(self.embedding(input))
        output, hidden = self.gru(embedded)

        return output, hidden
    
class BahdanauAttention(torch.nn.Module):
    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        self.Wa = torch.nn.Linear(hidden_size, hidden_size)
        self.Ua = torch.nn.Linear(hidden_size, hidden_size)
        self.Va = torch.nn.Linear(hidden_size, 1)

    def forward(self, query, keys):
        scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))
        weights = torch.nn.functional.softmax(scores, dim=-1)
        weights = weights.squeeze(3)
        keys = keys.squeeze(1)
        
        context = torch.bmm(weights, keys)
        context = context.sum(dim=1, keepdim=True)
        return context, weights

class AttnDecoderRNN(torch.nn.Module):
    def __init__(self, output_size, hidden_size, device, num_layers=1, dropout_p=0.2, hid_dropout_p = 0.5):
        super(AttnDecoderRNN, self).__init__()

        self.kwargs = {"output_size": output_size, "hidden_size": hidden_size, "device": device, "num_layers": num_layers, "dropout_p": dropout_p}
        self.device = device

        self.embedding = torch.nn.Embedding(output_size, hidden_size)
        self.attention = BahdanauAttention(hidden_size)
        self.gru = torch.nn.GRU(2 * hidden_size, hidden_size, num_layers=num_layers, dropout=hid_dropout_p, batch_first=True)
        self.out1 = torch.nn.Linear(hidden_size, hidden_size, device=device)
        self.out2 = torch.nn.Linear(hidden_size, output_size, device=device)
        self.dropout = torch.nn.Dropout(dropout_p)

    def forward(self, encoder_outputs, encoder_hidden, SOS_token , teacher_forcing_p=1, target_tensor=None):
        batch_size = encoder_outputs.size(0)
        
        decoder_input = torch.empty(size=[batch_size, 1], dtype=torch.long, device=self.device).fill_(SOS_token)
        decoder_hidden = encoder_hidden
        decoder_outputs = []
        attentions = []

        MAX_LENGTH = target_tensor.shape[1] if target_tensor is not None else 50

        for i in range(MAX_LENGTH):
            decoder_output, decoder_hidden, attn_weights = self.forward_step(
                decoder_input, decoder_hidden, encoder_outputs
            )
            decoder_outputs.append(decoder_output)
            attentions.append(attn_weights)

            teacher_forcing = random.random() < teacher_forcing_p

            if teacher_forcing and target_tensor is not None:
                # Teacher forcing: Feed the target as the next input
                decoder_input = target_tensor[:, i].unsqueeze(1) # Teacher forcing
            else:
                # Without teacher forcing: use its own predictions as the next input
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()  # detach from history as input

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = torch.nn.functional.log_softmax(decoder_outputs, dim=-1)
        attentions = torch.cat(attentions, dim=1)

        return decoder_outputs, decoder_hidden, attentions


    def forward_step(self, input, hidden, encoder_outputs):
        embedded =  self.dropout(self.embedding(input))
        query = hidden.permute(1, 0, 2)
        query = query.unsqueeze(2)
        encoder_outputs = encoder_outputs.unsqueeze(1)

        context, attn_weights = self.attention(query, encoder_outputs)

        input_gru = torch.cat((embedded, context), dim=2)

        output, hidden = self.gru(input_gru, hidden)

        output = self.out1(output)
        output = torch.relu(output) ** 2
        output = self.out2(output)

        return output, hidden, attn_weights
    

class Seq2Seq(torch.nn.Module):
    def __init__(self, input_size, output_size, hidden_size, device, num_layers=1, dropout_p=0.2, hid_dropout_p = 0.5):
        super(Seq2Seq, self).__init__()

        self.kwargs = {"input_size": input_size, "output_size": output_size, "hidden_size": hidden_size, "device": device, "num_layers": num_layers, "dropout_p": dropout_p}
        
        self.encoder = EncoderRNN(input_size=input_size, 
                                  hidden_size=hidden_size, 
                                  num_layers=num_layers, 
                                  dropout_p=dropout_p, 
                                  hid_dropout_p=hid_dropout_p)
        
        self.decoder = AttnDecoderRNN(output_size=output_size, 
                                      hidden_size=hidden_size, 
                                      device=device, 
                                      num_layers=num_layers, 
                                      dropout_p=dropout_p, 
                                      hid_dropout_p=hid_dropout_p)
        
    def forward(self, input_tensor, target_tensor, teacher_forcing_p=1):
        encoder_outputs, encoder_hidden = self.encoder(input_tensor)
        decoder_outputs, decoder_hidden, attentions = self.decoder(encoder_outputs, 
                                                                   encoder_hidden,
                                                                   SOS_token=1,
                                                                   teacher_forcing_p=teacher_forcing_p,
                                                                   target_tensor=target_tensor)
        return decoder_outputs, decoder_hidden, attentions
        
    def predict(self, input_tensor, SOS_token=1, EOS_token=2):
        self.encoder.eval()
        self.decoder.eval()
        
        with torch.no_grad():
            encoder_outputs, encoder_hidden = self.encoder(input_tensor)
            decoder_outputs, decoder_hidden, decoder_attn = self.decoder(encoder_outputs, encoder_hidden, SOS_token)

            decoder_outputs = decoder_outputs.squeeze(0)

            decoded_ids = decoder_outputs.argmax(dim=-1, keepdim=False)
            decoded_tokens = []
            for idx in decoded_ids:
                if idx.item() == EOS_token:
                    decoded_tokens.append(EOS_token)
                    break
                decoded_tokens.append(idx.item())

        return decoded_tokens