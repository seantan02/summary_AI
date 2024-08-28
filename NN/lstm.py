import torch
import random

class Encoder(torch.nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.embedding = torch.nn.Embedding(input_dim, embedding_dim)
        self.rnn = torch.nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=dropout, batch_first=True)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, src):
        # src = [batch size, src length]
        embedded = self.dropout(self.embedding(src))
        # embedded = [batch size, src length, embedding dim]
        outputs, (hidden, cell) = self.rnn(embedded)
        # outputs = [batch size, src length, hidden dim * n directions]
        # hidden = [n layers * n directions, batch size, hidden dim]
        # cell = [n layers * n directions, batch size, hidden dim]
        return hidden, cell
    

class Decoder(torch.nn.Module):
    def __init__(self, output_dim, embedding_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.embedding = torch.nn.Embedding(output_dim, embedding_dim)
        self.rnn = torch.nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=dropout, batch_first=True)
        self.fc_out = torch.nn.Linear(hidden_dim, output_dim)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, input, hidden, cell):
        # input = [batch size]
        # hidden = [n layers * n directions, batch size, hidden dim]
        # cell = [n layers * n directions, batch size, hidden dim]
        input = input.unsqueeze(1)
        # input = [batch size, 1]
        embedded = self.dropout(self.embedding(input))
        # embedded = [batch size, 1, embedding dim]
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        # output = [batch size, 1, hidden dim]
        # hidden = [n layers, batch size, hidden dim]
        # cell = [n layers, batch size, hidden dim]
        prediction = self.fc_out(output.squeeze(1))
        # prediction = [batch size, output dim]
        return prediction, hidden, cell
    

class Seq2Seq(torch.nn.Module):
    def __init__(self, input_size, output_size, input_embed_size, output_embed_size, hidden_size, device, num_layers=1, dropout_p=0.2, hid_dropout_p = 0.5):
        super().__init__()
        self.encoder = Encoder(input_size, input_embed_size, hidden_size, num_layers, dropout_p)
        self.decoder = Decoder(output_size, output_embed_size, hidden_size, num_layers, dropout_p)
        self.device = device

        self.kwargs = {"input_size": input_size, "output_size": output_size, "input_embed_size": input_embed_size, "output_embed_size": output_embed_size, "hidden_size": hidden_size, "device": device, "num_layers": num_layers, "dropout_p": dropout_p, "hid_dropout_p": hid_dropout_p}

        assert (
            self.encoder.hidden_dim == self.decoder.hidden_dim
        ), "Hidden dimensions of encoder and decoder must be equal!"
        assert (
            self.encoder.n_layers == self.decoder.n_layers
        ), "Encoder and decoder must have equal number of layers!"

    def forward(self, src, trg, teacher_forcing_p):
        # src = [batch size, src length]
        # trg = [batch size, trg length]
        # teacher_forcing_ratio is probability to use teacher forcing
        batch_size = trg.shape[0]
        trg_length = trg.shape[1]
        trg_vocab_size = self.decoder.output_dim
        # tensor to store decoder outputs
        outputs = torch.zeros(batch_size, trg_length, trg_vocab_size).to(self.device)
        # last hidden state of the encoder is used as the initial hidden state of the decoder
        hidden, cell = self.encoder(src)
        # first input to the decoder is the <sos> tokens
        input = trg[:, 0]
        for t in range(1, trg_length):
            # insert input token embedding, previous hidden and previous cell states
            # receive output tensor (predictions) and new hidden and cell states
            output, hidden, cell = self.decoder(input, hidden, cell)
            # place predictions in a tensor holding predictions for each token
            outputs[:, t] = output
            # decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_p
            # get the highest predicted token from our predictions
            top1 = output.argmax(1)
            # if teacher forcing, use actual next token as next input
            # if not, use predicted token
            input = trg[:, t] if teacher_force else top1
        return outputs

    def predict(self, src, SOS_token=1, EOS_token=2, max_len=50):
        # src = [1, src length]
        self.eval()
        with torch.no_grad():
            hidden, cell = self.encoder(src)
            input = torch.tensor([SOS_token]).to(self.device)
            outputs = []
            for _ in range(max_len):
                output, hidden, cell = self.decoder(input, hidden, cell)
                top1 = output.argmax(1)
                outputs.append(top1.item())
                if top1.item() == EOS_token:
                    break
                input = top1
        return outputs
