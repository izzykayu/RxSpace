from __future__ import print_function
import plac
import spacy
from pathlib import Path
import datetime
import glob
import re
import jsonlines
import random
from spacy.util import minibatch, compounding
today = datetime.datetime.today()
print(today)

def readBrat(input_file_name):
    with open(input_file_name + ".txt") as tf:
        text = tf.read()

    with open(input_file_name + ".ann") as an:
        entities = []
        other_entities = []
        for line in an.readlines():
            if not line.strip():
                continue
            concept_regex = "^(T\d+)\t(\w+)\s(\d+)\s(\d+)\t(.*)"
            match = re.search(concept_regex, line.strip())
            if match:
                groups = match.groups()
                entities.append((int(groups[2]), int(groups[3]), groups[1]))
            else:
                other_entities.append(line)

        return text, entities, other_entities

def write_out_jsonlines(TRAIN_DATA, outpath):
    with jsonlines.open(outpath, "w") as fout:
        fout.write_all(TRAIN_DATA)
    #     for DAT in TRAIN_DATA:
    #          fout.write(DAT)
    #
    # fout.close()

def make_all(pn_path, outpath):
    pn = glob.glob(pn_path)
    training_data = []
    with jsonlines.open(outpath, 'w') as writer:
        for tf in pn:
            name = tf.split(".txt")[0]
            basename = name.split("/")[-1]
            if (name is None) or (name == ""):
                continue
     # , other_entities
            text, entities, _  = readBrat(name)
            writer.write({'text': text, 'entities': entities, 'meta': {'id': basename}})
            training_data.append((text, {"entities": entities}))#, {"meta": {"id": basename}})) #, {"other_entities": other_entities}))
    return training_data

TRAIN_DATA = make_all("../bc2gm-corpus/standoff/test/*.txt", outpath= "./corpus/bc2gm_spacy_test.jsonl")
# write_out_jsonlines(TRAIN_DATA, "./corpus/bc2gm_spacy_test.jsonl")
# print(TRAIN_DATA)

@plac.annotations(
    model=("model name ", "option", "m", str),
  # input_file=("input spacy file", "option", "i", str),
    output_dir=("Optional output directory to save model", "option", "o", Path),
    n_iter=("Number of training iterations", "option", "n", int),
    )
def main(model=None, output_dir=None, n_iter=100):

    if model is not None:
        nlp = spacy.load(model)  # load existing spaCy model
        print("Loaded model '%s'" % model)
    else:
        nlp = spacy.blank("en")  # create blank Language class
        print("Created blank 'en_core_sci_md' model")

    if "ner" not in nlp.pipe_names:
        ner = nlp.create_pipe("ner")
        nlp.add_pipe(ner, last=True)
    # otherwise, get it so we can add labels
    else:
        ner = nlp.get_pipe("ner")

    # TRAIN_DATA = []
    # with jsonlines.open(input_file, 'r') as f_in:
    #     for line in f_in:
    #         TRAIN_DATA.append(line)

    for _, annotations in TRAIN_DATA:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
    with nlp.disable_pipes(*other_pipes):

        # training a new model
        if model is None:
            nlp.begin_training()
        for itn in range(n_iter):
            random.shuffle(TRAIN_DATA)
            losses = {}
            # using compounding matches
            batches = minibatch(TRAIN_DATA, size=compounding(4.0, 32.0, 1.001))
            # switch back to 32 from 16

            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(texts,  # batch t
                    annotations,  # batch a
                    drop=0.3, # add Adam params here
                    losses=losses,)
            print("Losses", losses)


    if output_dir is not None:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()
        nlp.to_disk(output_dir)
        print("Saved model to thus directory: ",output_dir)

        print("... Loading from the direcroty: ",output_dir)
        nlp2 = spacy.load(output_dir)
        print(" --predicting on dataset train--")
        for text, _ in TRAIN_DATA:
            doc = nlp2(text)
            print("predicting text")
            print("Entities", [(ent.text, ent.label_) for ent in doc.ents])

if __name__ == '__main__':
    plac.call(main)