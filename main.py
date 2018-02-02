from code.EncoderDecoder import *
from code.lang import *
import torch
import random
RANDOM_SEED = 10000

class Controller:
    def __init__(
        self,
        vi_embedding_size = 50,
        en_embedding_size = 50,
        encoder_hidden_unit_size = 1000,
        decoder_hidden_unit_size = 1000,
        dataset_path = 'dataset/',
        allow_gpu = True,
        train_percentage = 1.0,
        test_percentage = 1.0,
        limit_vocab = True,
        default_vocab_limit = 2000,
        vi_to_en = True,
        debug = False
    ):
        self.en_train_filepath = dataset_path + 'train/train.en'
        self.vi_train_filepath = dataset_path + 'train/train.vi'

        self.en_test_filepath = dataset_path + 'test/tst2012.en'
        self.vi_test_filepath = dataset_path + 'test/tst2012.vi'

        self.vocab = {
                'EN':Vocab(
                    dataset_path + 'vocab/vocab.en',
                    size_limit = default_vocab_limit,
                    use_limit = limit_vocab
                ),
                'VI':Vocab(
                    dataset_path + 'vocab/vocab.en',
                    size_limit = default_vocab_limit,
                    use_limit = limit_vocab
                ),
            }

        self.use_gpu = allow_gpu and torch.cuda.is_available()
        self.vi_to_en = vi_to_en
        self.en_train_sentences = self.process_sentences(
                open(self.en_train_filepath).read().split('\n'),
                'EN'
            )

        self.vi_train_sentences = self.process_sentences(
                open(self.vi_train_filepath).read().split('\n'),
                'VI'
            )

        self.en_test_sentences = self.process_sentences(
                open(self.en_test_filepath).read().split('\n'),
                'EN'
            )

        self.vi_test_sentences = self.process_sentences(
                open(self.vi_test_filepath).read().split('\n'),
                'VI'
            )
        if debug:
            self.en_train_sentences = self.en_train_sentences[:10]
            self.vi_train_sentences = self.vi_train_sentences[:10]
            self.en_test_sentences = self.en_test_sentences[:10]
            self.vi_test_sentences = self.vi_test_sentences[:10]
        self.embeddings = EmbeddingVector(
                self.vocab,
                en_dim = en_embedding_size,
                vi_dim = vi_embedding_size,
                use_gpu = self.use_gpu
            )

        if self.vi_to_en:
            self.encoder = Encoder(
                    vi_embedding_size,
                    encoder_hidden_unit_size,
                    self.embeddings,
                    use_gpu = self.use_gpu
                )

            self.decoder = Decoder(
                    decoder_hidden_unit_size,
                    en_embedding_size,
                    self.embeddings,
                    use_gpu = self.use_gpu
                )
        else:
            self.encoder = Encoder(
                    en_embedding_size,
                    encoder_hidden_unit_size,
                    self.embeddings,
                    use_gpu = self.use_gpu
                )
            self.decoder = Decoder(
                    decoder_hidden_unit_size,
                    vi_embedding_size,
                    self.embeddings,
                    use_gpu = self.use_gpu
                )
        if self.use_gpu:
            self.encoder = self.encoder.cuda()
            self.decoder = self.decoder.cuda()
        self.check_controller()


    def check_controller(self):
        print("Checking data...")
        assert len(self.en_train_sentences) == len(self.vi_train_sentences), \
            'unequal number of EN VI sentences in training data'

        assert len(self.en_test_sentences) == len(self.vi_test_sentences), \
            'unequal number of EN VI sentences in testing data'

        print("LGTM :)")

    def run(
        self,
        num_epochs = 500,
        start_learning_rate = 0.01,
        teacher_ratio = 0.5
    ):
        print_every = 1
        train_loss = 0
        test_loss = 0
        encoder_optimizer = torch.optim.SGD(self.encoder.parameters(), lr=start_learning_rate)
        decoder_optimizer = torch.optim.SGD(self.decoder.parameters(), lr=start_learning_rate)
        criterion = torch.nn.NLLLoss()
        enc_scheduler = torch.optim.lr_scheduler.StepLR(encoder_optimizer, step_size=100, gamma=0.1)
        dec_scheduler = torch.optim.lr_scheduler.StepLR(decoder_optimizer, step_size=100, gamma=0.1)
        if self.vi_to_en:
            train_input_sentences = self.vi_train_sentences
            train_output_sentences = self.en_train_sentences
            test_input_sentences = self.vi_test_sentences
            test_output_sentences = self.en_test_sentences
        else:
            train_input_sentences = self.en_train_sentences
            train_output_sentences = self.vi_train_sentences
            test_input_sentences = self.en_test_sentences
            test_output_sentences = self.vi_test_sentences
        for epoch in range(1, num_epochs + 1):
            enc_scheduler.step()
            dec_scheduler.step()
            for in_sent, out_sent in zip(train_input_sentences, train_output_sentences):
                input_variable = in_sent
                target_variable = out_sent
                loss = self.train(input_variable, target_variable, encoder_optimizer, decoder_optimizer, criterion)
                train_loss += loss


            for in_sent, out_sent in zip(test_input_sentences, test_output_sentences):
                input_variable = in_sent
                target_variable = out_sent
                loss, _ = self.evaluate(input_variable, target_variable, criterion)
                test_loss += loss

            if epoch % print_every == 0:
                train_loss_avg = train_loss / print_every
                train_loss = 0
                test_loss_avg = test_loss / print_every
                test_loss = 0

                print('Epoch: %d -- Training Loss: %.4f -- Testing Loss: %.4f' % (epoch, train_loss_avg, test_loss_avg))

    def train(self, input_variable, target_variable, encoder_optimizer, decoder_optimizer, criterion):
        encoder_hidden = self.encoder.initHidden()

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        input_length = len(input_variable)
        target_variable = target_variable.split(' ')
        target_length = len(target_variable)

        loss = 0
        encoder_output, encoder_hidden = self.encoder(input_variable, encoder_hidden)
        encoder_output = encoder_output.cuda() if self.use_gpu else encoder_output
        decoder_input = '<s>'
        decoder_hidden = self.decoder.initHidden()
        teacher_forcing_ratio = 0.5
        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
        if use_teacher_forcing:
            for di in range(target_length):
                decoder_output, decoder_hidden = self.decoder(
                    encoder_output, decoder_input, decoder_hidden)
                loss += criterion(decoder_output, self.process_variable(target_variable[di]))
                decoder_input = target_variable[di]  # Teacher forcing
        else:
            for di in range(target_length):
                decoder_output, decoder_hidden = self.decoder(
                    encoder_output, decoder_input, decoder_hidden)
                topv, topi = decoder_output.data.topk(1)
                ni = topi[0][0]
                decoder_input = self.vocab['EN'].get_word(ni)
                loss += criterion(decoder_output, self.process_variable(target_variable[di]))
                if decoder_input == '</s>':
                    break

        loss.backward()

        encoder_optimizer.step()
        decoder_optimizer.step()

        return loss.data[0] / target_length

    def evaluate(self, input_variable, target_variable, criterion):
        encoder_hidden = self.encoder.initHidden()

        input_length = len(input_variable)
        target_variable = target_variable.split(' ')
        target_length = len(target_variable)
        answer = []
        loss = 0

        encoder_output, encoder_hidden = self.encoder(input_variable, encoder_hidden)
        encoder_output = encoder_output.cuda() if self.use_gpu else encoder_output
        decoder_input = '<s>'
        decoder_hidden = self.decoder.initHidden()
        for di in range(target_length):
            decoder_output, decoder_hidden = self.decoder(
                encoder_output, decoder_input, decoder_hidden)
            topv, topi = decoder_output.data.topk(1)
            ni = topi[0][0]
            decoder_input = self.vocab['EN'].get_word(ni)
            loss += criterion(decoder_output, self.process_variable(target_variable[di]))
            answer.append(decoder_input)
            if decoder_input == '</s>':
                break
        return loss.data[0] / target_length, answer

    def process_variable(self, var):
	    a = self.vocab['EN'].get_key(var)
	    tensor = torch.LongTensor([a])
	    a = Variable(tensor)
	    if self.use_gpu:
	        return a.cuda()
	    return a

    def process_sentences(self, sents, lang):
        lst = []
        for sent in sents:
            if self.vi_to_en:
                if lang == 'EN':
                    a = sent + ' </s>'
                elif lang == 'VI':
                    a = '<s> ' + sent + ' </s>'
            else:
                if lang == 'VI':
                    a = sent + ' </s>'
                elif lang == 'EN':
                    a = '<s> ' + sent + ' </s>'
            lst.append(a)
        return lst
if __name__ == '__main__':
    torch.manual_seed(RANDOM_SEED)
    model_controller = Controller(
            allow_gpu = True,
            debug=True
        )
    model_controller.run()
