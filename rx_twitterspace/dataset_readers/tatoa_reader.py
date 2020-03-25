import logging
from typing import Dict
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import LabelField, TextField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers.character_tokenizer import CharacterTokenizer
from allennlp.data.vocabulary import Vocabulary
from overrides import overrides

logger = logging.getLogger(__name__)

DatasetReader.register("tatoeba_sentence_reader")
class TatoebaSentenceReader(DatasetReader):
    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 lazy=False):
        super().__init__(lazy=lazy)
        self.tokenizer = CharacterTokenizer()
        self.token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}


    @overrides
    def text_to_instance(self, tokens, label=None):
        fields = {}

        fields['tokens'] = TextField(tokens, self.token_indexers)
        if label:
            fields['label'] = LabelField(label)

        return Instance(fields)


    @overrides
    def _read(self, file_path: str):
        file_path = cached_path(file_path)
        with open(file_path, "r") as text_file:
            for line in text_file:
                lang_id, sent = line.rstrip().split('\t')

                tokens = self.tokenizer.tokenize(sent)

                yield self.text_to_instance(tokens, lang_id)

