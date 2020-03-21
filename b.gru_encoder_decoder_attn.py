import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from data_prep import WikiSQL_S2S
from ipdb import launch_ipdb_on_exception
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SOS_token = 0
EOS_token = 1


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, embedding_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size

        self.embedding = nn.Embedding(input_size, embedding_size)
        self.gru = nn.GRU(embedding_size, hidden_size)

    def forward(self, input, hidden):
        sequence_len = input.size()[0]
        embedded = self.embedding(input)
        # GRU expects (seq_len, batch_size, input_size)
        embedded = embedded.view(sequence_len, -1, self.embedding_size)

        output = embedded
        output, hidden = self.gru(output, hidden)

        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p, max_length, embedding_size):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.embedding_size = embedding_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


def train(encoder, decoder, eoptim, doptim, loss_fn, train_loader):

    # training
    total_loss, total_accu = 0, 0
    for x, y in tqdm(train_loader):
        loss, accu, encoder_hidden = _train(
            input_tensor=x.to(device),
            target_tensor=y.to(device),
            encoder=encoder,
            decoder=decoder,
            encoder_optimizer=eoptim,
            decoder_optimizer=doptim,
            criterion=loss_fn,
        )
        total_loss += loss
        total_accu += accu

        norm_loss = total_loss / len(train_loader)
        norm_accu = total_accu / len(train_loader)

    return norm_loss, norm_accu, encoder_hidden


def _train(input_tensor,
           target_tensor,
           encoder,
           decoder,
           encoder_optimizer,
           decoder_optimizer,
           criterion):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    num_inputs = input_tensor.size(0)
    target_length = target_tensor.size(1)

    encoder_outputs = torch.zeros(
        decoder.max_length,
        encoder.hidden_size,
        device=device
    )

    loss = 0

    for ei in range(num_inputs):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_token]], device=device)

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True  # if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input

        decoded_sequence = []
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)

            loss += criterion(decoder_output, target_tensor[0][di].unsqueeze(0))
            decoder_input = target_tensor[0][di]  # Teacher forcing

            topv, topi = decoder_output.topk(1)
            decoded_sequence.append(topi)

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    accu = (target_tensor == torch.as_tensor(decoded_sequence).to(device)).sum()
    norm_accu = accu.item() / target_length
    norm_loss = loss.item() / target_length

    return norm_loss, norm_accu, encoder_hidden


def evaluate(encoder, encoder_hidden, decoder, eval_loader):
    encoder.eval()
    decoder.eval()

    total_loss, total_accu = 0, 0
    for x, y in tqdm(eval_loader):
        loss, accu = _evaluate(
            encoder=encoder,
            encoder_hidden=encoder_hidden,
            decoder=decoder,
            x=x.to(device),
            y=y.to(device)
        )

        total_loss += loss
        total_accu += accu

    norm_loss = total_loss / len(eval_loader)
    norm_accu = total_accu / len(eval_loader)

    encoder.train()
    decoder.train()

    return norm_loss, norm_accu


def _evaluate(encoder, encoder_hidden, decoder, x, y):

    num_inputs = x.size(0)
    target_length = y.size(1)

    loss = 0
    encoder_outputs = torch.zeros(
        decoder.max_length,
        encoder.hidden_size,
        device=device
    )

    for ei in range(num_inputs):
        encoder_output, encoder_hidden = encoder(x[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_token]], device=device)

    decoder_hidden = encoder_hidden

    decoded_sequence = []
    for di in range(target_length):
        decoder_output, decoder_hidden, decoder_attention = decoder(
            decoder_input, decoder_hidden, encoder_outputs)

        loss += criterion(decoder_output, y[0][di].unsqueeze(0))
        decoder_input = y[0][di]  # Teacher forcing

        topv, topi = decoder_output.topk(1)
        decoded_sequence.append(topi)

    # Calculating unit NLLLoss
    accu = (y == torch.as_tensor(decoded_sequence).to(device)).sum()
    norm_accu = accu.item() / target_length
    norm_loss = loss.item() / target_length

    return norm_loss, norm_accu


if __name__ == "__main__":

    # Dataset portion
    INCLUSION_RATIO = 0.10
    # with launch_ipdb_on_exception():
    train_datset = WikiSQL_S2S(
        data_dir="./data",
        portion="train",
        reduced_set_perc=INCLUSION_RATIO
    )
    test_dataset = WikiSQL_S2S(
        data_dir="./data",
        portion="test",
        reduced_set_perc=INCLUSION_RATIO
    )
    print("WikiSQL dataset loaded.")

    # Name this expt for logging results in a seperate folder
    EXPT_NAME = "attn"
    # hidden repr size and GRU N hidden units
    NUM_HIDDEN_UNITS = 64
    # Possible input vocab size
    NUM_IN_VOCAB = train_datset.in_tokenizer.get_vocab_size()
    # Possible output vocab size
    NUM_OUT_VOCAB = train_datset.out_tokenizer.get_vocab_size()
    # Embedding representation size
    EMBEDDING_UNITS = 50
    # Maximum sequence length for any input in X
    MAX_LENGTH = train_datset.MAX_SEQ_LEN
    # Attention layer dropout proba
    ATTN_DROPOUT = 0.1
    # Learning rate of encoder & decoder optimizers
    LEARNING_RATE = 0.001
    # Number of epochs
    EPOCHS = 300

    writer = SummaryWriter(log_dir=f'./data/log/{EXPT_NAME}')

    encoder = EncoderRNN(
        input_size=NUM_IN_VOCAB,
        hidden_size=NUM_HIDDEN_UNITS,
        embedding_size=EMBEDDING_UNITS
    ).to(device)
    decoder = AttnDecoderRNN(
        hidden_size=NUM_HIDDEN_UNITS,
        output_size=NUM_OUT_VOCAB,
        dropout_p=ATTN_DROPOUT,
        max_length=MAX_LENGTH,
        embedding_size=EMBEDDING_UNITS
    ).to(device)

    train_iterator = torch.utils.data.DataLoader(train_datset, batch_size=1)
    test_iterator = torch.utils.data.DataLoader(test_dataset, batch_size=1)
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=LEARNING_RATE)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=LEARNING_RATE)
    criterion = nn.NLLLoss()

    for epoch in tqdm(range(1, EPOCHS + 1)):
        train_loss, train_acc, encoder_hidden = train(
            encoder=encoder,
            decoder=decoder,
            eoptim=encoder_optimizer,
            doptim=decoder_optimizer,
            loss_fn=criterion,
            train_loader=train_iterator
        )

        test_loss, test_acc = evaluate(
            encoder=encoder,
            encoder_hidden=encoder_hidden,
            decoder=decoder,
            eval_loader=test_iterator
        )

        writer.add_scalars("SQL2Cypher/accuracy", {
            "train": train_acc,
            "eval": test_acc
        }, epoch)
        writer.add_scalars("SQL2Cypher/loss", {
            "train": train_loss,
            "eval": test_loss
        }, epoch)
