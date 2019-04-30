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

def model_config(parser):
    parser.add_argument('--update_bert_opt',  default=0, type=int)
    parser.add_argument('--multi_gpu_on', action='store_true')
    parser.add_argument('--mem_cum_type', type=str, default='simple',
                        help='bilinear/simple/defualt')
    parser.add_argument('--answer_num_turn', type=int, default=5)
    parser.add_argument('--answer_mem_drop_p', type=float, default=0.1)
    parser.add_argument('--answer_att_hidden_size', type=int, default=128)
    parser.add_argument('--answer_att_type', type=str, default='bilinear',
                        help='bilinear/simple/defualt')
    parser.add_argument('--answer_rnn_type', type=str, default='gru',
                        help='rnn/gru/lstm')
    parser.add_argument('--answer_sum_att_type', type=str, default='bilinear',
                        help='bilinear/simple/defualt')
    parser.add_argument('--answer_merge_opt', type=int, default=1)
    parser.add_argument('--answer_mem_type', type=int, default=1)
    parser.add_argument('--answer_dropout_p', type=float, default=0.1)
    parser.add_argument('--answer_weight_norm_on', action='store_true')
    parser.add_argument('--dump_state_on', action='store_true')
    parser.add_argument('--answer_opt', type=int, default=0, help='0,1')
    parser.add_argument('--label_size', type=str, default='3')
    parser.add_argument('--mtl_opt', type=int, default=0)
    parser.add_argument('--ratio', type=float, default=0)
    parser.add_argument('--mix_opt', type=int, default=0)
    parser.add_argument('--max_seq_len', type=int, default=512)
    parser.add_argument('--init_ratio', type=float, default=1)
    return parser

def data_config(parser):
    parser.add_argument('--log_file', default='mt-dnn-pred.log', help='path for log file.')
    parser.add_argument("--init_checkpoint", default='mt_dnn/bert_model_base.pt', type=str)
    parser.add_argument('--input_file', type=str, default=None)
    parser.add_argument('--pw_tasks', default='qnnli', type=str, help="comma separated values "
                                                                    "of pairwise tasks")
    return parser

def predict_config(parser):
    parser.add_argument('--task', required=True, type=str)
    parser.add_argument('--evaluate', action="store_true")
    parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available(),
                        help='whether to use GPU acceleration.')
    parser.add_argument('--log_per_updates', type=int, default=500)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--batch_size_eval', type=int, default=8)
    parser.add_argument('--optimizer', default='adamax',
                        help='supported optimizer: adamax, sgd, adadelta, adam')
    parser.add_argument('--grad_clipping', type=float, default=0)
    parser.add_argument('--global_grad_clipping', type=float, default=1.0)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--momentum', type=float, default=0)
    parser.add_argument('--warmup', type=float, default=0.1)
    parser.add_argument('--warmup_schedule', type=str, default='warmup_linear')

    parser.add_argument('--vb_dropout', action='store_false')
    parser.add_argument('--dropout_p', type=float, default=0.1)
    parser.add_argument('--dropout_w', type=float, default=0.000)
    parser.add_argument('--bert_dropout_p', type=float, default=0.1)

    # EMA
    parser.add_argument('--ema_opt', type=int, default=0)
    parser.add_argument('--ema_gamma', type=float, default=0.995)

    # scheduler
    parser.add_argument('--have_lr_scheduler', dest='have_lr_scheduler', action='store_false')
    parser.add_argument('--multi_step_lr', type=str, default='10,20,30')
    parser.add_argument('--freeze_layers', type=int, default=-1)
    parser.add_argument('--embedding_opt', type=int, default=0)
    parser.add_argument('--lr_gamma', type=float, default=0.5)
    parser.add_argument('--bert_l2norm', type=float, default=0.0)
    parser.add_argument('--scheduler_type', type=str, default='ms', help='ms/rop/exp')
    parser.add_argument('--output_file', default='out.txt')
    parser.add_argument('--seed', type=int, default=2018,
                        help='random seed for data shuffling, embedding init, etc.')
    parser.add_argument('--task_config_path', type=str, default='configs/tasks_config.json')

    return parser


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
            
            

def predict(args,task_name, evaluate = False):
    """Predicts from an input file and writes to an output file the results.
    args is an instance of ArgumentParser"""

    logger.info('Starting preprocessing')


    input_data = process_input_file(args.input_file,
                                    args.max_seq_len,
                                    task_name,
                                    evaluate)

    opt = vars(args)
    
    batch_size = args.batch_size
    
    pw_task = task_name in args.pw_tasks 

    dataset = BatchGen(input_data,
                    batch_size=batch_size,
                    dropout_w=args.dropout_w,
                    gpu=args.cuda,
                    task_id=0,
                    maxlen=args.max_seq_len,
                    pairwise=pw_task,
                    data_type=DATA_TYPE[task_name],
                    task_type=TASK_TYPE[task_name],
                    is_train = False)

    tasks_config = {}
    if os.path.exists(args.task_config_path):
        with open(args.task_config_path, 'r') as reader:
            tasks_config = json.loads(reader.read())

    opt['answer_opt']  = [generate_decoder_opt(task_name, opt['answer_opt'])]

    dropout_p = args.dropout_p
    if tasks_config and task_name in tasks_config:
        dropout_p = tasks_config[task_name]
    opt['tasks_dropout_p'] = [dropout_p]

    num_labels = DATA_META[task_name]
    args.label_size = f"{num_labels}"


    model_path = args.init_checkpoint
    state_dict = None

    assert os.path.exists(model_path), f"Model folder doesn't exist: f{model_path}"
    
    state_dict = torch.load(model_path)
    config = state_dict['config']
    config['attention_probs_dropout_prob'] = args.bert_dropout_p
    config['hidden_dropout_prob'] = args.bert_dropout_p

    opt.update(config)

    model = MTDNNModel(opt, state_dict=state_dict)
    model.eval()

    logger.info('\n{}\n'.format(model.network))

    if args.cuda:
        model.cuda()

    predictions = []
    start = datetime.now()
    with torch.no_grad():
        metrics, predictions, scores, golds, ids =\
                         eval_model(model, dataset, dataset=task_name,
                                            use_cuda=args.cuda,
                                            with_label=evaluate)

    logger.info(f"Prediction Loop finished in time: {datetime.now()-start}")

    logger.info(predictions)
    logger.info(metrics)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser = data_config(parser)
    parser = model_config(parser)
    parser = predict_config(parser)
    args = parser.parse_args()
    args.pw_tasks = list(set([pw for pw in args.pw_tasks.split(',') if len(pw.strip()) > 0]))
    pprint(args)

    set_environment(args.seed, args.cuda)
    log_path = args.log_file
    logger =  create_logger(__name__, to_disk=True, log_file=log_path)
    logger.info(args.answer_opt)

    predict(args,args.task,evaluate=args.evaluate)