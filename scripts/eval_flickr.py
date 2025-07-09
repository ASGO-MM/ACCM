from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice
import pandas as pd
import json
import sys
import ipdb
import argparse

class Evaluator:
    def __init__(self) -> None:
        self.tokenizer = PTBTokenizer()
        self.scorer_list = [
            (Cider(), "CIDEr"),
            # (Meteor(), "METEOR")
        ]
        self.evaluation_report = {}

    def do_the_thing(self, golden_reference, candidate_reference):
        golden_reference = self.tokenizer.tokenize(golden_reference)
        candidate_reference = self.tokenizer.tokenize(candidate_reference)
        
        # From this point, some variables are named as in the original code
        # I have no idea why they name like these
        # The original code: https://github.com/salaniz/pycocoevalcap/blob/a24f74c408c918f1f4ec34e9514bc8a76ce41ffd/eval.py#L51-L63
        for scorer, method in self.scorer_list:
            score, scores = scorer.compute_score(golden_reference, candidate_reference)
            if isinstance(method, list):
                for sc, scs, m in zip(score, scores, method):
                    self.evaluation_report[m] = sc
            else:
                self.evaluation_report[method] = score


parser = argparse.ArgumentParser()
parser.add_argument("--answer_file", type=str)
parser.add_argument("--anno_file", type=str)
args = parser.parse_args()

# df = pd.read_csv('../Datasets/Flickr30k/flickr_annotations_30k.csv')
# df = df[df['split'] == 'test']

f_anno = open(args.anno_file, 'r')
annotations = [json.loads(x)['text'] for x in f_anno.readlines()]
golden_reference = annotations

f = open(args.answer_file, 'r')
outputs = [json.loads(x)['text'] for x in f.readlines()]
candidate_reference = outputs
#ipdb.set_trace()
# for i, x in enumerate(df.iloc):
#     s = x['raw'][2:-2].replace('"','').split(',')
#     golden_reference.append(s)
#     # print(outputs[i])
#     candidate_reference.append(outputs[i])

# golden_reference = {k: [{'caption': x} for x in v] for k, v in enumerate(golden_reference)}
golden_reference = {k: [{'caption': v}] for k, v in enumerate(golden_reference)}

candidate_reference = {k: [{'caption': v}] for k, v in enumerate(candidate_reference)}
#ipdb.set_trace()
# breakpoint()
evaluator = Evaluator()
evaluator.do_the_thing(golden_reference, candidate_reference)

print(evaluator.evaluation_report)