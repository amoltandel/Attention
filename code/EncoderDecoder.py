import torch
from torch.autograd import Variable
import torch.nn.functional as F

class Encoder(torch.nn.Module):
    def __init__(self,
        input_embedding_size,
        hidden_unit_size,
        embeddings,
        lang = 'VI',
        use_gpu = False,
    ):
        super(Encoder, self).__init__()
        self.input_size = input_embedding_size
        self.hidden_size = hidden_unit_size
        self.embedding = embeddings
        self.lang = lang
        self.use_gpu = use_gpu
        self.gru = torch.nn.GRU(
            self.input_size,
            self.hidden_size,
            batch_first = True,
            bidirectional = True
        )

    def forward(self, input, hidden):
        embedded = self.embedding.get_embeddings(
                input,
                lang = self.lang
            ).unsqueeze(0)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        result = Variable(torch.zeros(2, 1, self.hidden_size))
        if self.use_gpu:
            return result.cuda()
        else:
            return result
class Decoder(torch.nn.Module):
    def __init__(
        self,
        hidden_size,
        output_size,
        embeddings,
        lang = 'EN',
        dropout_p = 0.1,
        use_gpu = False
    ):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.embedding = embeddings
        self.dropout_p = dropout_p
        self.dropout = torch.nn.Dropout(self.dropout_p)
        self.encoder_weights = torch.nn.Linear(hidden_size * 2, 1)
        self.previous_hidden_weights = torch.nn.Linear(hidden_size, 1)
        self.use_gpu = use_gpu
        self.gru = torch.nn.GRU(
            hidden_size * 2,
            hidden_size,
            batch_first = True
        )
        self.lang = lang
        self.out = torch.nn.Linear(
            hidden_size,
            self.embedding.embeddings[lang].num_embeddings
        ) # vocab length
        self.softmax = torch.nn.LogSoftmax(dim = 1)

    def forward(self, encoder_output, word_input, previous_hidden_state):
        word_embedding = self.embedding.get_embeddings(
                word_input,
                lang=self.lang
            ).unsqueeze(0)
        word_embedding = self.dropout(word_embedding)
        output1 = self.encoder_weights(encoder_output.squeeze(0))
        output2 = self.previous_hidden_weights(previous_hidden_state.squeeze(0))
        alpha = torch.nn.Softmax(dim = 0)(output1 + output2)
        context = (encoder_output.squeeze(0) * alpha).\
            sum(0).unsqueeze(0).unsqueeze(0)
        context = F.relu(context)
        output, hidden = self.gru(context, previous_hidden_state)
        output = F.log_softmax(self.out(output[0]), dim = 1)
        return output, hidden

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if self.use_gpu:
            return result.cuda()
        return result
