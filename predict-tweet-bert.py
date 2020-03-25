
from typing import Iterator, List, Dict
import torch
import torch.optim as optim
import numpy as np
from allennlp.data import Instance
from allennlp.data.fields import TextField, SequenceLabelField
from allennlp.data.dataset_readers import DatasetReader
from allennlp.common.file_utils import cached_path
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder, PytorchSeq2SeqWrapper
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.data.iterators import BucketIterator
from allennlp.training.trainer import Trainer
from allennlp.predictors import SentenceTaggerPredictor
from rx_twitterspace.dataset_readers import ClassificationDatasetReader
torch.manual_seed(1)
from allennlp.data.token_indexers import PretrainedBertIndexer
from allennlp.modules.token_embedders import BertEmbedder

bert = PretrainedBertIndexer(do_lowercase= False,
                pretrained_model = "scibert_scivocab_uncased/vocab.txt",
                use_starting_offsets=True)
vocb = Vocabulary.from_files(

"saved-models/model1/vocabulary"
    )


bert_embedder = BertEmbedder.bert_model(type="bert-pretrained",
                                        bert_model="scibert_scivocab_uncased/weights.hdf5")



# Pass in the ElmoTokenEmbedder instance instead
word_embeddings = BasicTextFieldEmbedder({"tokens": bert_embedder})

# The dimension of the ELMo embedding will be 2 x [size of LSTM hidden states]
elmo_embedding_dim = 256

reader = ClassificationDatasetReader(tokenizer=bert_embedder, token_indexers=bert)
predictor = SentenceTaggerPredictor(model, dataset_reader=reader)
tag_logits = predictor.predict("The dog ate the apple")['tag_logits']
tag_ids = np.argmax(tag_logits, axis=-1)


print([model.vocab.get_token_from_index(i, 'labels') for i in tag_ids])
# Here's how to save the model.

# And here's how to reload the model.
model2 = LstmTagger(word_embeddings, lstm, vocab2)
with open("/tmp/model.th", 'rb') as f:
    model2.load_state_dict(torch.load(f))

predictor2 = SentenceTaggerPredictor(model2, dataset_reader=reader)
tag_logits2 = predictor2.predict("The dog ate the apple")['tag_logits']
np.testing.assert_array_almost_equal(tag_logits2, tag_logits)