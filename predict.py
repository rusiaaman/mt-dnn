# Copyright (c) Microsoft. All rights reserved.
"""Prediction and evaluation from input_file. To evaulate pass
--evaluate"""

import argparse
import json
import os
import random
from datetime import datetime
from pprint import pprint

import numpy as np
import torch
from tqdm import tqdm

from data_utils.glue_utils import submit, eval_model
from data_utils.label_map import DATA_META, GLOBAL_MAP, DATA_TYPE, DATA_SWAP, TASK_TYPE, generate_decoder_opt
from data_utils.log_wrapper import create_logger
from data_utils.utils import set_environment
from mt_dnn.batcher import BatchGen
from mt_dnn.model import MTDNNModel
from pytorch_pretrained_bert.modeling import BertModel
from pytorch_pretrained_bert.modeling import BertConfig

from prepro import _truncate_seq_pair,bert_tokenizer


def data_config(parser):
    parser.add_argument("--checkpoint_dir", default='mt_dnn', type=str)
    parser.add_argument('--input_file', type=str, default=None)
    return parser

def predict_config(parser):
    parser.add_argument('--task', required=True, type=str)
    parser.add_argument('--evaluate', action="store_true")
    parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available(),
                        help='whether to use GPU acceleration.')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--output_file', default='out.txt')
    return parser


def get_model(directory,task):
    logger.info('Loading model')
    model_path = os.path.join(directory,task,"model.pt")

    state_dict = None

    assert os.path.exists(model_path), f"Model folder doesn't exist: f{model_path}"
    
    state_dict = torch.load(model_path)
    opt = state_dict['config']


    model = MTDNNModel(opt, state_dict=state_dict)
    model.eval()


    return model,opt
class Predictor:

    def __init__(self,checkpoint_dir,task,batch_size,cuda=False):

        self.task_name = task

        self.model,self.opt = get_model(checkpoint_dir,self.task_name)

        self.cuda = cuda

        if self.cuda:
            self.model.cuda()

        self.batch_size = batch_size


    @staticmethod
    def process_input_file(input_file,max_seq_len,task_name,evaluate=False):

        if not input_file or not os.path.exists(input_file):
            raise Exception("Please provide a valid input_file. "
                                f"Doesn't exists {input_file}")
        
        assert task_name in DATA_TYPE and task_name in DATA_META,\
                                    (f"{task_name} not a valid type"
                                    f". Allowed options:"
                                    f" {set(DATA_META).union(set(DATA_TYPE))}")

        if task_name in ["sst","cola"]:
            """Sentence classification tasks"""
            def get_features(vals,i):
                uid = str(i)
                assert (not evaluate and len(vals) == 1) or\
                        (evaluate and len(vals) == 2), \
                                    (f"{task_name} should have single sentence "
                                        " for prediction and a label if evaluation"
                                    f"sentence as input. line number: {i}")

                sentence = vals[0]
                premise = bert_tokenizer.tokenize(sentence)
                if len(premise) >  max_seq_len - 3:
                    premise = premise[:max_seq_len - 3] 
                input_ids = bert_tokenizer.convert_tokens_to_ids(
                                                ['[CLS]'] + premise + ['[SEP]'])
                type_ids = [0] * ( len(premise) + 2)
                features = {'uid': uid, 'token_id': input_ids, 
                                    'type_id': type_ids, 'task': task_name}
                if evaluate:
                    label = int(vals[-1])
                    features.update({"label": label})
                return features
            
        elif task_name in ["qnli"]:
            """For pairwise tasks"""
            raise NotImplementedError("QNLI is not yet supported")
            #TODO: check ruid and olabel to write down for qnli
        else:
            """Single premise and hypothesis case"""
            def get_features(vals,i):
                uid = str(i)
                assert (not evaluate and len(vals) == 2) or\
                        (evaluate and len(vals) == 3), \
                                    (f"{task_name} should have two"
                                    f"sentences as input and a label if evaluating"
                                    f" . line number: {i}")

                premise = bert_tokenizer.tokenize(vals[1])
                hypothesis = bert_tokenizer.tokenize(vals[2])
                _truncate_seq_pair(premise, hypothesis, max_seq_len - 3)
                input_ids =bert_tokenizer.convert_tokens_to_ids(['[CLS]'] + hypothesis + ['[SEP]'] + premise + ['[SEP]'])
                type_ids = [0] * ( len(hypothesis) + 2) + [1] * (len(premise) + 1)
                features = {'uid': uid, 'token_id': input_ids, 
                        'type_id': type_ids, 'task': task_name}
                if evaluate:
                    label = int(vals[-1])
                    features.update({"label": label})

                return features

        with open(input_file) as f:
            all_features = []
            logger.info(f"Reading file {input_file}")
            for i,line in enumerate(tqdm(f)):
                
                
                vals = line.split('\t')

                if len(vals)<1:
                    raise ValueError("Invalid line in input_file: "
                        f"\n{line}\nline number: {i}")

                all_features.append(get_features(vals,i))

        return all_features
                
                

    def predict_from_file(self,input_file,evaluate = False):
        """Predicts from an input file and writes to an output file the results.
        args is an instance of ArgumentParser"""

        logger.info('Starting preprocessing')


        input_data = Predictor.process_input_file(input_file,
                                        self.opt['max_seq_len'],
                                        self.task_name,
                                        evaluate)

        
        pw_task = self.task_name in self.opt['pw_tasks'] 

        dataset = BatchGen(input_data,
                        batch_size=self.batch_size,
                        dropout_w=self.opt['dropout_w'],
                        gpu=self.cuda,
                        task_id=0,
                        maxlen=self.opt['max_seq_len'],
                        pairwise=pw_task,
                        data_type=DATA_TYPE[self.task_name],
                        task_type=TASK_TYPE[self.task_name],
                        is_train = False)


        num_labels = DATA_META[self.task_name]

        

        logger.info('\n{}\n'.format(self.model.network))


        predictions = []
        start = datetime.now()
        with torch.no_grad():
            metrics, predictions, scores, golds, ids =\
                             eval_model(self.model, dataset, dataset=self.task_name,
                                                use_cuda=self.cuda,
                                                with_label=evaluate)

        logger.info(f"Prediction Loop finished in time: {datetime.now()-start}")

        logger.info(predictions)
        logger.info(metrics)

        return metrics, predictions, scores, golds, ids

    


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser = data_config(parser)
    parser = predict_config(parser)
    args = parser.parse_args()
    pprint(args)

    logger =  create_logger(__name__)

    predictor = Predictor(args.checkpoint_dir,args.task,
                    args.batch_size,args.cuda)

    predictor.predict_from_file(args.input_file,evaluate=args.evaluate)