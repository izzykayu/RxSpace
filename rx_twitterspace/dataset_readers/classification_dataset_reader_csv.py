""" Data reader for AllenNLP """


from typing import Dict, List, Any
import logging

import csv
from overrides import overrides
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import LabelField, TextField, MultiLabelField, ListField, ArrayField, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Tokenizer, WordTokenizer, CharacterTokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("classification_csv_dataset_reader")
class ClassificationCSVDatasetReader(DatasetReader):
    """
    Text classification data reader

    The data is assumed to be in jsonlines format
    each line is a json-dict with the following keys: 'text', 'label', 'metadata'
    'metadata' is optional and only used for passing metadata to the model
    """
    def __init__(self,
                 lazy: bool = False,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 ) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer or CharacterTokenizer()
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

    @overrides
    def _read(self, file_path):
        with open(file_path, 'r') as f_in:
            reader = csv.DictReader(f_in, delimiter=',')
            for json_object in reader:
                yield self.text_to_instance(
                    text=json_object.get('unprocessed_text'),
                    label=json_object.get('class').strip(),
                    metadata=json_object.get('tweetid')
                )

    @overrides
    def text_to_instance(self,
                         text: str = 'unprocessed_text',
                         label: str = 'class',
                         metadata: Any = None) -> Instance:  # type: ignore
        text_tokens = self._tokenizer.tokenize(text)
        fields = {
            'text': TextField(text_tokens, self._token_indexers),
        }
        if label is not None:
            fields['label'] = LabelField(label)

        # if metadata:
        #     fields['metadata'] = MetadataField(metadata)
        return Instance(fields)
