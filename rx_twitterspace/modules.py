import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np

__all__ = [
    'ConvLanguageModel',
    'BiLanguageModel',
    'CRF',
    'BiLMCRFTagger',
    'HiddenReprRelationshipClassifier',
    'HiddenReprTagRelationshipClassifier',
    'CharCNN',
    'Highway',
    'ConcatCharFuser',
    'GatingCharFuser',
    'ElmoProjector',
]


class ConvLanguageModel(nn.Module):
    def __init__(self,
                 word_field,
                 hidden_dim,
                 num_layers=1,
                 embedding_dim=None,
                 packed=True,
                 char_cnn=None,
                 char_field=None):
        '''
        A Convolutional Language Model.

        One can either call forward() to predict the next/previous word given
        the word sequence, or call get_hidden() to retrieve the hidden states
        only.  The latter is useful for plugging into a bigger model.

        NOTE: Only works on torch 0.4.1 or newer.

        Parameters
        word_field    : A torchtext.data.Field instance for words.  The field
                        should have its vocabulary built.
        hidden_dim    : Size of hidden states.
        embedding_dim : Embedding size.  If the word vocabulary has initialized
                        its vectors then this argument is ignored.
        packed        : Use packed RNN instead of reversing the sequences
                        manually (not yet implemented)
        char_cnn      : Character CNN module for computing the character-level
                        embeddings (not yet implemented)
        char_field    : A torchtext.data.NestedField instance for characters from
                        each word (not yet implemented)
        '''
        nn.Module.__init__(self)

        word_vocab = word_field.vocab
        self.lrelu = nn.LeakyReLU()
        self.word_field = word_field
        self.hidden_dim = hidden_dim
        self.char_cnn = char_cnn
        self.char_field = char_field
        self.vocab_size = len(word_vocab)
        self.embedding_dim = (word_vocab.vectors.shape[1]
                              if word_vocab.vectors is not None else
                              embedding_dim)
        self.num_layers = 6
        self.cnns = nn.ModuleList()
        output_dim = hidden_dim
        self.output_dim = output_dim
        # Word embeddings, optionally initialize with given vectors
        self.emb = nn.Embedding(
            self.vocab_size,
            self.embedding_dim,
            padding_idx=word_vocab.stoi[word_field.pad_token],
        )

        if word_vocab.vectors is not None:
            self.emb.weight.data.copy_(word_vocab.vectors)

        # I could have used the multilayer torch.nn.LSTM but maybe
        # we need outputs at intermediate layers (e.g. in ELMo).
        k = 3
        s = 1
        d = 2
        effective_k = 3 + 2 * 1
        p = 2
        cnn = nn.Conv1d(
            in_channels=self.embedding_dim,
            out_channels=hidden_dim,
            kernel_size=k,
            stride=s,
            padding=p,
            dilation=d,
            groups=1,
            bias=True)
        self.cnns.append(cnn)
        for i in range(self.num_layers - 2):
            # input_dim = self.embedding_dim if i == 0 else hidden_dim
            cnn = nn.Conv1d(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                kernel_size=k,
                stride=s,
                padding=p,
                dilation=d,
                groups=1,
                bias=True)
            self.cnns.append(cnn)
        cnn = nn.Conv1d(
            in_channels=hidden_dim,
            out_channels=output_dim,
            kernel_size=k,
            stride=s,
            padding=p,
            dilation=d,
            groups=1,
            bias=True)
        self.cnns.append(cnn)

    def get_hidden(self,
                   word_idx,
                   word_len,
                   max_word_len=None,
                   char_idx=None,
                   char_len=None,
                   max_char_len=None):
        '''
        Pass the word indices into the BiLM and get the hidden states of all
        layers.

        Parameters:
        word_idx     : Word indices.  LongTensor of shape
                       (max_word_len, batch_size)
                       If using packed RNN, the sequences should be sorted in
                       decreasing length.
        word_len     : Length of each sentence by words.
        max_word_len : Always pad the result sequence to length max_word_len.
                       Useful for running with nn.DataParallel.

        Returns:
        A hidden state tensor of shape
        (num_layers, max_word_len, batch_size, 2, hidden_dim)
        The forward and backward direction is 0 and 1 respectively, following
        the convention in nn.LSTM.
        '''
        max_word_len = max_word_len or word_idx.shape[0]
        batch_size = word_idx.shape[1]
        x = self.emb(word_idx)

        hs = []
        h = pack_padded_sequence(x, word_len)
        h = torch.transpose(x, 1, 2)
        h = torch.transpose(h, 0, 2)
        state = None
        for idx, cnn in enumerate(self.cnns):
            h = cnn(h)
            h = self.lrelu(h)
            hs.append(h)

        hs = torch.stack(hs, 0)

        return hs

    def forward(self,
                word_idx,
                word_len,
                max_word_len=None,
                char_idx=None,
                char_len=None,
                max_char_len=None):
        hs = self.get_hidden(
            word_idx,
            word_len,
            max_word_len=max_word_len,
            char_idx=char_idx,
            char_len=char_len,
            max_char_len=max_char_len,
        )
        h = hs[-1]

        # TODO: predict next/previous word !!
        return h


class Highway(nn.Module):
    def __init__(self, features, nonlinearity=F.leaky_relu):
        nn.Module.__init__(self)

        self.T = nn.Linear(features, features)
        self.H = nn.Linear(features, features)
        self.nonlinearity = nonlinearity

    def forward(self, x):
        t = torch.sigmoid(self.T(x))
        h = self.nonlinearity(self.H(x))
        return t * h + (1 - t) * x


class CharCNN(nn.Module):
    def __init__(self,
                 char_field,
                 hidden_dim,
                 num_layers=1,
                 embedding_dim=16,
                 highway=None):
        nn.Module.__init__(self)

        char_vocab = char_field.vocab
        self.lrelu = nn.LeakyReLU()
        self.char_field = char_field
        self.hidden_dim = hidden_dim
        self.vocab_size = len(char_vocab)
        self.embedding_dim = (char_vocab.vectors.shape[1]
                              if char_vocab.vectors is not None else
                              embedding_dim)
        self.num_layers = num_layers
        self.cnns = nn.ModuleList()
        output_dim = hidden_dim
        self.output_dim = output_dim
        # Word embeddings, optionally initialize with given vectors
        self.emb = nn.Embedding(
            self.vocab_size,
            self.embedding_dim,
            padding_idx=char_vocab.stoi[char_field.pad_token],
        )

        if char_vocab.vectors is not None:
            self.emb.weight.data.copy_(char_vocab.vectors)

        if num_layers == 1:
            cnn = nn.Conv1d(
                in_channels=self.embedding_dim,
                out_channels=output_dim,
                kernel_size=5,
                padding=2,
                bias=True)
            self.cnns.append(cnn)
        else:   # deep
            cnn = nn.Conv1d(
                in_channels=self.embedding_dim,
                out_channels=hidden_dim,
                kernel_size=3,
                padding=1,
                bias=True)
            self.cnns.append(cnn)
            for _ in range(self.num_layers - 2):
                cnn = nn.Conv1d(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    kernel_size=3,
                    padding=1,
                    bias=True)
                self.cnns.append(cnn)
            cnn = nn.Conv1d(
                in_channels=hidden_dim,
                out_channels=output_dim,
                kernel_size=3,
                padding=1,
                bias=True)
            self.cnns.append(cnn)

        self.highway = highway

    def get_hidden(self, char_idx, num_chars):
        x = self.emb(char_idx)
        hs = []
        bs, num_words, word_len, char_emb_dim = x.size()
        x = x.view(-1, word_len, char_emb_dim)
        h = torch.transpose(x, 1, 2)
        for cnn in self.cnns:
            h = cnn(h)
            h = self.lrelu(h)
            hs.append(
                torch.transpose(h, 1, 2).view(bs, num_words, word_len, -1))

        hs = torch.stack(hs, 0)

        return hs

    def forward(self, char_idx, num_chars):
        hs = self.get_hidden(char_idx, num_chars)
        h = hs[-1]

        y = h.max(2)[0]

        if self.highway is not None:
            y = self.highway(y)

        return y


class ConcatCharFuser(nn.Module):
    def __init__(self, word_dim, char_dim, output_dim=None):
        nn.Module.__init__(self)

        self.word_dim = word_dim
        self.char_dim = char_dim
        self.output_dim = word_dim + char_dim

    def forward(self, word_emb, char_emb):
        return torch.cat([word_emb, char_emb], -1)


class GatingCharFuser(nn.Module):
    def __init__(self, word_dim, char_dim, output_dim=None, nonlinearity=torch.tanh):
        nn.Module.__init__(self)

        self.word_dim = word_dim
        self.char_dim = char_dim
        self.output_dim = output_dim
        self.word_proj = nn.Linear(word_dim, output_dim)
        self.char_proj = nn.Linear(char_dim, output_dim)
        self.word_gate = nn.Linear(word_dim, 1)
        ## can someone help me understand this part sometime :) - izzy
        self.nonlinearity = nonlinearity

    def forward(self, word_emb, char_emb):
        g = torch.sigmoid(self.word_gate(word_emb))
        w = self.nonlinearity(self.word_proj(word_emb))
        c = self.nonlinearity(self.char_proj(char_emb))
        return g * w + (1 - g) * c


class BiLanguageModel(nn.Module):
    def __init__(self,
                 word_field,
                 hidden_dim,
                 num_layers=1,
                 embedding_dim=None,
                 packed=True,
                 char_cnn=None,
                 char_field=None,
                 char_fuser_class=None):
        '''
        A Bidirectional Language Model.

        One can either call forward() to predict the next/previous word given
        the word sequence, or call get_hidden() to retrieve the hidden states
        only.  The latter is useful for plugging into a bigger model.

        NOTE: Only works on torch 0.4.1 or newer.

        Parameters
        word_field    : A torchtext.data.Field instance for words.  The field
                        should have its vocabulary built.
        hidden_dim    : Size of hidden states.
        embedding_dim : Embedding size.  If the word vocabulary has initialized
                        its vectors then this argument is ignored.
        packed        : Use packed RNN instead of reversing the sequences
                        manually (not yet implemented)
        char_cnn      : Character CNN module for computing the character-level
                        embeddings (not yet implemented)
        char_field    : A torchtext.data.NestedField instance for characters from
                        each word (not yet implemented)
        '''
        nn.Module.__init__(self)

        word_vocab = word_field.vocab

        self.word_field = word_field
        self.hidden_dim = hidden_dim
        self.char_cnn = char_cnn
        self.char_field = char_field
        self.vocab_size = len(word_vocab)
        self.embedding_dim = (word_vocab.vectors.shape[1]
                              if word_vocab.vectors is not None else
                              embedding_dim)
        self.num_layers = num_layers
        self.rnns = nn.ModuleList()

        # Word embeddings, optionally initialize with given vectors
        self.emb = nn.Embedding(
            self.vocab_size,
            self.embedding_dim,
            padding_idx=word_vocab.stoi[word_field.pad_token],
        )
        if word_vocab.vectors is not None:
            self.emb.weight.data.copy_(word_vocab.vectors)

        if char_cnn is not None:
            assert char_fuser_class is not None
            self.char_fuser = char_fuser_class(
                    self.embedding_dim, self.char_cnn.output_dim, hidden_dim)
            first_input_dim = self.char_fuser.output_dim
        else:
            first_input_dim = self.embedding_dim

        # I could have used the multilayer torch.nn.LSTM but maybe
        # we need outputs at intermediate layers (e.g. in ELMo).
        for i in range(num_layers):
            input_dim = first_input_dim if i == 0 else hidden_dim * 2
            rnn = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                bidirectional=True)
            self.rnns.append(rnn)

    def get_hidden(self,
                   word_idx,
                   word_len,
                   max_word_len=None,
                   char_idx=None,
                   char_len=None,
                   num_chars=None,
                   max_char_len=None):
        '''
        Pass the word indices into the BiLM and get the hidden states of all
        layers.

        Parameters:
        word_idx     : Word indices.  LongTensor of shape
                       (max_word_len, batch_size)
                       If using packed RNN, the sequences should be sorted in
                       decreasing length.
        word_len     : Length of each sentence by words.
        max_word_len : Always pad the result sequence to length max_word_len.
                       Useful for running with nn.DataParallel.

        Returns:
        A hidden state tensor of shape
        (num_layers, max_word_len, batch_size, 2, hidden_dim)
        The forward and backward direction is 0 and 1 respectively, following
        the convention in nn.LSTM.
        '''
        max_word_len = max_word_len or word_idx.shape[0]
        batch_size = word_idx.shape[1]

        x = self.emb(word_idx)
        if self.char_cnn is not None:
            x_charcnn = self.char_cnn(char_idx, num_chars)
            x_charcnn = torch.transpose(x_charcnn, 1, 0)
            x = self.char_fuser(x, x_charcnn)
        hs = []
        h = pack_padded_sequence(x, word_len)
        state = None
        for rnn in self.rnns:
            h, state = rnn(h, state)
            hs.append(h)

        hs = torch.stack([(pad_packed_sequence(h, total_length=max_word_len)[0]
                           .view(max_word_len, batch_size, 2, self.hidden_dim))
                          for h in hs], 0)

        return hs

    def predict_word(self, hs):
        return None

    def forward(self,
                word_idx,
                word_len,
                max_word_len=None,
                char_idx=None,
                char_len=None,
                num_chars=None,
                max_char_len=None):
        hs = self.get_hidden(
            word_idx,
            word_len,
            max_word_len=max_word_len,
            char_idx=char_idx,
            char_len=char_len,
            num_chars=num_chars,
            max_char_len=max_char_len,
        )

        return self.predict_word(hs)


class CRF(nn.Module):
    # TODO: add more inputs into CRF

    def __init__(self, n_classes=None, transitions=None,
                 transitions_mask=None):
        '''
        A trainable CRF module

        Parameters:
        n_classes   : number of classes.  Must provide if transitions is None
        transitions : transition matrix, if available.  Rows correspond to
                      previous class and columns correspond to next class
        '''
        nn.Module.__init__(self)

        self.n_classes = n_classes or transitions.shape[0]
        self.transitions = (nn.Parameter(torch.randn(n_classes, n_classes)) if
                            transitions is None else nn.Parameter(transitions))
        self.register_buffer(
            'transitions_mask',
            torch.ones_like(self.transitions).byte()
            if transitions_mask is None else transitions_mask)

    def decode(self, scores, lengths):
        '''
        Find the maximum sum of scores.

        Parameters:
        scores  : a FloatTensor of shape (seq_len, batch_size, n_classes)
        lengths : a LongTensor of shape (batch_size) containing length of
                  each score sequence

        Returns:
        max_score : the maximum sum of scores for each example, a FloatTensor
                    with shape (batch_size)
        path      : a LongTensor of shape (seq_len, batch_size), indicating
                    the class assignment of each sequence for the maximum
                    sum of scores.  Any value beyond the length of the
                    sequence should be ignored.
        '''
        seq_len, batch_size, n_classes = scores.shape
        zero = scores.new(1).zero_()
        transitions = torch.where(
            self.transitions_mask,
            self.transitions,
            zero - 1000,
        )
        # each element contains the maximum score so far if the class of
        # the current element is the corresponding class
        max_score_if = scores.new(batch_size, n_classes).zero_()
        # preceding class for the maximum score
        prec = []

        for t in range(seq_len):
            if t == 0:
                max_score_if += scores[t]
            else:
                t_mask = (t < lengths)
                # c[:, i, j] = maximum score if y[t-1] = i and y[t] = j
                #            = max_score[i] + T[i, j] + s[t][j]
                # also set the transition and scores to be 0 if t
                # exceeds the length of the sequence
                # c is next value contributions to max_score_if
                # max_score_if[bs, i, :] is the max score until this point with y[t-1] being i
                # max_score_if[bs, i, j] is the max score until y such that y[t-1] and y[t] is j
                c = max_score_if[:, :, None] + torch.where(
                    t_mask[:, None, None],
                    transitions[None, :, :] + scores[t, :, None, :],
                    zero,
                )
                max_score_if, current_prec = c.max(1)
                prec.append(current_prec)

        # find the maximum score of each example and backtrack using prec
        max_score, choice = max_score_if.max(1)
        path = [choice]
        for cur_prec in reversed(prec):
            choice = cur_prec.gather(1, choice[:, None])[:, 0]
            path.insert(0, choice)
        path = torch.stack(path, 0)

        return max_score, path

    def log_likelihood(self, scores, target, lengths):
        '''
        Find the log-likelihood of a given target sequence on the linear-chain
        CRF defined by scores and transition matrix.

        Parameters:
        scores  : a FloatTensor of shape (seq_len, batch_size, n_classes)
        target  : a LongTensor of shape (seq_len, batch_size)
        lengths : a LongTensor of shape (batch_size) containing length of
                  each score/target sequence

        Returns:
        target_score   : a FloatTensor with shape (batch_size)
        log_likelihood : a FloatTensor with shape (batch_size)
        '''
        seq_len, batch_size, n_classes = scores.shape
        zero = scores.new(1).zero_()
        transitions = torch.where(
            self.transitions_mask,
            self.transitions,
            zero - 1000,
        )
        # log-normalization constant so far.  The idea is the same as Viterbi
        logZ_if = scores.new(batch_size, n_classes).zero_()
        X = scores.new(batch_size).zero_()

        for t in range(seq_len):
            if t == 0:
                logZ_if += scores[t]
                X += scores[t].gather(1, target[t, :, None])[:, 0]
            else:
                t_mask = (t < lengths)
                c = logZ_if[:, :, None] + \
                    transitions[None, :, :] + \
                    scores[t, :, None, :]
                # Halt logZ computation if t goes beyond the sequence
                logZ_if = torch.where(t_mask[:, None], c.logsumexp(1), logZ_if)
                X += torch.where(
                    t_mask,
                    transitions[target[t - 1], target[t]] + scores[t].gather(
                        1, target[t, :, None])[:, 0], zero)

        logZ = logZ_if.logsumexp(1)

        return X, X - logZ

    def forward(self, scores, target, lengths):
        return self.log_likelihood(scores, target, lengths)


class BiLMCRFTagger(nn.Module):
    def __init__(self, bilm, crf, proj=None, hidden_extractor=None):
        '''
        Bidirectional language model CRF tagger.

        Takes in a BiLanguageModel instance and a CRF instance, and optionally
        a projector instance which transforms hidden states to tag scores.

        If no projector is provided, the tagger will concatenate the forward
        and backward hidden states at the last layer on each time step, and
        project the concatenation to a linear layer to get the scores.

        If you wish to use your own projector, it should take in two arguments:
        (a) a hidden state tensor
            (num_bilm_layers, max_seq_len, batch_size, 2, hidden_dim)
        (b) a sequence length tensor (batch_size)
        And outputs a score tensor
        (max_seq_len, batch_size, n_classes)
        '''
        nn.Module.__init__(self)

        self.bilm = bilm
        self.crf = crf
        self.proj = proj
        self.hidden_extractor = hidden_extractor
        if proj is None:
            self.default_proj = nn.Linear(bilm.hidden_dim * 2, crf.n_classes)

    def forward(self,
                word_idx,
                word_len,
                target=None,
                max_word_len=None,
                char_idx=None,
                char_len=None,
                max_char_len=None):
        '''
        The module operates in two modes, depending on whether the target
        sequence is provided.

        Parameters
        word_idx     : Word indices.  LongTensor of shape
                       (max_word_len, batch_size)
                       If using packed RNN, the sequences should be sorted in
                       decreasing length.
        word_len     : Length of each sentence by words.
        target       : The target sequence.  Can be None or LongTensor of shape
                       (max_word_len, batch_size)
        max_word_len : Always pad the result sequence to length max_word_len.
                       Useful for running with nn.DataParallel.

        Returns

        If the target is provided, the forward pass returns the score and
        log-likelihood of each example in the batch.
        target_score   : a FloatTensor with shape (batch_size)
        log_likelihood : a FloatTensor with shape (batch_size)

        If the target is not provided, the forward pass returns
        max_score : the maximum sum of scores for each example, a FloatTensor
                    with shape (batch_size)
        path      : a LongTensor of shape (seq_len, batch_size), indicating
                    the class assignment of each sequence for the maximum
                    sum of scores.  Any value beyond the length of the
                    sequence should be ignored.

        Attributes change:
        hs : the hidden states of BiLM
        '''
        seq_len, batch_size = word_idx.shape
        hs = self.bilm.get_hidden(
            word_idx,
            word_len,
            max_word_len=max_word_len,
            char_idx=char_idx,
            char_len=char_len,
            max_char_len=max_char_len)

        if self.hidden_extractor is not None:
            h = self.hidden_extractor(hs)
        else:
            h = hs[-1]

        self.hs = hs
        self.h = h

        if self.proj is None:
            h = h.view(seq_len * batch_size, -1)
            s = self.default_proj(h)
            s = s.view(seq_len, batch_size, -1)
        else:
            s = self.proj(h)

        if target is None:
            return self.crf.decode(s, word_len)
        else:
            return self.crf.log_likelihood(s, target, word_len)


class HiddenReprRelationshipClassifier(nn.Module):
    def __init__(self, in_dims, hidden_dims, tags):
        nn.Module.__init__(self)
        self.W1 = nn.Sequential(
                nn.Linear(in_dims, hidden_dims),
                nn.LeakyReLU(),
                nn.Linear(hidden_dims, hidden_dims),
                )
        self.W2 = nn.Sequential(
                nn.Linear(in_dims, hidden_dims),
                nn.LeakyReLU(),
                nn.Linear(hidden_dims, hidden_dims),
                )

    def forward(self, h1, h2, tag_idx):
        '''
        Classifies whether two hidden states are related to each other
        Parameters:
        h1: (batch_size, hidden_dim)
        h2: (batch_size, hidden_dim)

        Returns:
        s: (batch_size) for score.
        '''
        h1 = self.W1(h1)
        h2 = self.W2(h2)
        return (h1[:, None, :] @ h2[:, :, None])[:, 0, 0]


class HiddenReprTagRelationshipClassifier(nn.Module):
    def __init__(self, in_dims, hidden_dims, tags):
        nn.Module.__init__(self)
        self.tags = {t: i for i, t in enumerate(tags)}
        self.tag_emb = nn.Embedding(len(tags), hidden_dims)
        self.W1 = nn.Sequential(
                nn.Linear(in_dims, hidden_dims),
                nn.LeakyReLU(),
                nn.Linear(hidden_dims, hidden_dims),
                )
        self.W2 = nn.Sequential(
                nn.Linear(in_dims + hidden_dims, hidden_dims),
                nn.LeakyReLU(),
                nn.Linear(hidden_dims, hidden_dims),
                )

    def forward(self, h1, h2, tags):
        tag_idx = torch.LongTensor([self.tags[t] for t in tags]).to(device=h1.device)
        h1 = self.W1(h1)
        h2 = self.W2(torch.cat([h2, self.tag_emb(tag_idx)], 1))
        return (h1[:, None, :] @ h2[:, :, None])[:, 0, 0]


class ElmoProjector(nn.Module):
    def __init__(self, layers_in, hidden_dim, n_classes):
        nn.Module.__init__(self)
        self.layers_in = layers_in
        self.n_classes = n_classes
        self.W = nn.Parameter(torch.randn(layers_in))

    def get_hidden(self, hs):
        w = F.softmax(self.W)
        layers, seq_len, bs, bidir_2, dimensions = hs.shape
        hs_weighted = hs * w.view(-1, 1, 1, 1, 1)
        hs_weighted = hs_weighted.sum(0)
        proj_in = hs_weighted.view(seq_len * bs, bidir_2 * dimensions)
        return proj_in.view(seq_len, bs, -1)

    def forward(self, hs):
        return self.get_hidden(hs)
