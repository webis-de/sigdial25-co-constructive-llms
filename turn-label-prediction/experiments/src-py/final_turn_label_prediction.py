import sys
import os
#import wandb

import transformers
import datasets
import argparse
import torch
import json
from pathlib import Path
from datasets import load_dataset, Dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification, default_data_collator,
                          DebertaV2ForSequenceClassification, DebertaV2Tokenizer,
                          TrainingArguments, Trainer, AutoConfig)
from transformers.modeling_outputs import SequenceClassifierOutput
from torch.utils.data import DataLoader

import numpy as np
import pandas as pd

from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from tabulate import tabulate
from glob import glob

from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler(random_state=123)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def majority_class(df):
    topics = df.topic.unique()
    for topic in topics:
        training_df = df[df.topic != topic]
        #compute the majority class for each label
        l = len(df[df.topic == topic])
        df.loc[df.topic == topic, 'topic_func_maj_pred'] = [training_df.topic_func_label.mode()] * l
        df.loc[df.topic == topic, 'dlg_act_maj_pred']    = [training_df.dlg_act_label.mode()] * l
        df.loc[df.topic == topic, 'exp_act_maj_pred']    = [training_df.exp_act_label.mode()] * l
    
    return df

def eval_preds(df, models_names, gt_clms, pred_clms):
    results_table = []
    for label in zip(gt_clms, pred_clms, models_names):
        ground_truths = df[label[0]].tolist()
        predictions   = df[label[1]].tolist()
        model_name = label[2]
        
        class_names = df[label[0]].unique()

        prc_scores = precision_score(ground_truths, predictions, average=None, labels=class_names)
        rec_scores = recall_score(ground_truths, predictions, average=None, labels=class_names)
        f1_scores  = f1_score(ground_truths, predictions, average=None, labels=class_names)
        
        macro_prc_scores = precision_score(predictions, ground_truths, average='macro', labels=class_names)
        macro_rec_scores = recall_score(predictions, ground_truths, average='macro', labels=class_names)
        macro_f1 = f1_score(predictions, ground_truths, average='macro', labels=class_names)
        
        scores ={}
        for i, c in enumerate(class_names):
            scores[c] = {'prec': round(prc_scores[i],2), 'recall': round(rec_scores[i],2), 'f1': round(f1_scores[i],2)}
        
        scores['Macro AVG.'] = {'prec': round(macro_prc_scores,2), 'recall': round(macro_rec_scores,2), 'f1': round(macro_f1,2)}
        
        results_table.append([model_name, label[0], scores])
    
    return results_table

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.argmax(-1)
    f1score = f1_score(predictions, labels, average='macro')
    return {'f1-score': f1score}

def train_model(output_dir, train_ds, valid_ds, test_ds, num_labels , model_name='bert-base-uncased', num_train_epochs=5, eval_steps=500, lr=2e-6, batch_size=4):

    args = TrainingArguments(
        output_dir= output_dir,
        eval_strategy = "epoch",
        save_strategy = "epoch",
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_train_epochs,
        weight_decay=0.01,
        eval_steps=eval_steps,
        load_best_model_at_end=True,
        metric_for_best_model='f1-score',
        report_to="none"
    )
    print('Loading {}'.format(model_name))
    if "deberta-v3" in model_name:
        model = DebertaV2ForSequenceClassification.from_pretrained(f"/bigwork/nhwpficl/hf_models/{model_name}", num_labels=num_labels).to(device)
        tokenizer = DebertaV2Tokenizer.from_pretrained(f"/bigwork/nhwpficl/hf_models/{model_name}")
    else:
        model = AutoModelForSequenceClassification.from_pretrained(f"/bigwork/nhwpficl/hf_models/{model_name}", num_labels=num_labels).to(device)
        tokenizer = AutoTokenizer.from_pretrained(f"/bigwork/nhwpficl/hf_models/{model_name}")

    multi_trainer =Trainer(
        model,
        args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        compute_metrics=lambda x: compute_metrics(x),
        processing_class=tokenizer
    )
    
    multi_trainer.train()
    
    model.save_pretrained(output_dir)
    eval_results = multi_trainer.evaluate(test_ds)
    
    return model, eval_results

def get_train_valid_splits(df, size=0.2):
    datasets = df.ds.unique()
    
    total_training_topics = []
    total_valid_topics = []
    for ds in datasets:
        topics = df[df.ds == ds].topic.unique()
        train_topics, valid_topics = train_test_split(topics, shuffle=False, test_size=size)
        total_training_topics += train_topics.tolist()
        total_valid_topics += valid_topics.tolist()
    
    return total_training_topics, total_valid_topics

def run_fold_experiment(train_df, test_df, label_clm, input_clm, output_dir, num_labels, model_name='bert-base-uncased', num_train_epochs=5, 
                   eval_steps=500, lr=2e-6, batch_size=4, val_size=0.2):
    
    config = AutoConfig.from_pretrained(f"/bigwork/nhwpficl/hf_models/{model_name}")
    
    if "deberta-v3" in model_name:
        tokenizer = DebertaV2Tokenizer.from_pretrained(f"/bigwork/nhwpficl/hf_models/{model_name}")
        max_length = config.max_position_embeddings * 2
    else:
        tokenizer = AutoTokenizer.from_pretrained(f"/bigwork/nhwpficl/hf_models/{model_name}")
        max_length = config.max_position_embeddings - 2
        

    #balance the data
    training_df, y = ros.fit_resample(train_df, train_df['labels'])
    training_df['labels'] = y

    train_ds = Dataset.from_pandas(training_df)
    valid_ds = Dataset.from_pandas(test_df)
    test_ds  = Dataset.from_pandas(test_df)
    
    sample_text = train_ds[input_clm][0]['text']
    encoded = tokenizer(sample_text)
    print(sample_text)
    print(tokenizer.convert_ids_to_tokens(encoded['input_ids']))
    
    train_ds = train_ds.map(lambda examples: tokenizer([x['text'] for x in examples[input_clm]], padding='max_length', max_length=max_length), batched=True)
    valid_ds = valid_ds.map(lambda examples: tokenizer([x['text'] for x in examples[input_clm]], padding='max_length', max_length=max_length), batched=True)
    test_ds  = test_ds.map(lambda examples: tokenizer([x['text'] for x in examples[input_clm]], padding='max_length', max_length=max_length), batched=True)
    
    model , eval_results = train_model(output_dir, train_ds, valid_ds, test_ds, num_labels, model_name=model_name, num_train_epochs=num_train_epochs, eval_steps=eval_steps, lr=lr, batch_size=batch_size)

    model.save_pretrained('{}/best_model'.format(output_dir))
    test_ds.to_json('{}/test_set.json'.format(output_dir))
    json.dump(eval_results, open('{}/eval_results.json'.format(output_dir), 'w'))

    return eval_results

def run_experiment(df, folds_dict, label_clm, input_clm, output_dir, model_name='bert-base-uncased', num_train_epochs=5, eval_steps=500, lr=2e-6, batch_size=4, val_size=0.2):
    all_eval_results = []
    
    df['labels'] = df[label_clm].apply(lambda x: int(x[2:4])-1) #making labels parasable as integers
    
    num_labels = df['labels'].nunique()
    
    train_df = df[df.topic.isin(folds_dict['train']['5lvls'] + folds_dict['train']['eli5'])]
    test_df  = df[df.topic.isin(folds_dict['test']['5lvls']  + folds_dict['test']['eli5'])]
    
    print(len(train_df))
    print(len(test_df))

    eval_results = run_fold_experiment(train_df, test_df, label_clm, input_clm, output_dir, model_name=model_name,  num_labels = num_labels, num_train_epochs=num_train_epochs, 
                        eval_steps=eval_steps, lr=lr, batch_size=batch_size, val_size=val_size)
    print(eval_results)

    all_eval_results.append(eval_results)
        
    return all_eval_results

def load_ds(ds_path, model_name):
    df = pd.read_pickle(ds_path)

    #Aligning the 5-levels labels to eli5 ones
        
    #'(D06) To answer - Other' -> '(D06) Answer - Other'
    #'(D07) To provide agreement statement' -> '(D07) Agreement'
    #'(D08) To provide disagreement statement' -> '(D08) Disagreement'
    #'(D10) Other' -> '(D09) Other'
    #'(D09) To provide informing statement' -> (D10) To provide informing statement
    
    
    # (E10) Other -> (E09) Other 
    # (E09) Introducing Extraneous Information -> (E10) Introducing Extraneous Information
    
    df['exp_act_label'] = df.exp_act_label.apply(lambda x: '(E10) Other' if x == '(E09) Other' else x)
    df['exp_act_label'] = df.exp_act_label.apply(lambda x: '(E09) Introducing Extraneous Information' if x == '(E10) Introducing Extraneous Information' else x)

    df['dlg_act_label'] = df.dlg_act_label.apply(lambda x: '(D09) Other' if x == '(D10) Other' else x)
    df['dlg_act_label'] = df.dlg_act_label.apply(lambda x: '(D10) To provide informing statement' if x == '(D09) To provide informing statement' else x)
    
    df['dlg_act_label'] = df.dlg_act_label.apply(lambda x: '(D06) Answer - Other' if x == '(D06) To answer - Other' else x)
    df['dlg_act_label'] = df.dlg_act_label.apply(lambda x: '(D07) Agreement' if x == '(D07) To provide agreement statement' else x)
    df['dlg_act_label'] = df.dlg_act_label.apply(lambda x: '(D08) Disagreement' if x == '(D08) To provide disagreement statement' else x)
    
    tokenizer = AutoTokenizer.from_pretrained(f"/bigwork/nhwpficl/hf_models/{model_name}")
    sep_token = tokenizer.sep_token

    df['turn_text_with_topic'] = df.apply(lambda row: {
                                        'author': row['turn_text']['author'], 
                                        'text'  : row['topic'].replace('_', ' ') + f' {sep_token} ' +  row['turn_text']['text']
                                       } ,axis=1)

    return df

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Train Models for Quality Prediction.')
    #parser.add_argument('label_clm', type=str)
    parser.add_argument('--output_path', type=str, default="../../data/final-turn-label-models")
    parser.add_argument('--ds', type=str, default=all)
    parser.add_argument('--num_train_epochs', type=int, default=10)
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--input_clm', type=str, required=False, default="turn_text_with_topic")
    parser.add_argument('--label_clm', type=str, required=False)
    
    
    args = parser.parse_args()
    
    #Loading and preparing data
    fivelvls_annotation_df = load_ds('../../data/five_levels_ds/annotation-results/MACE-measure/final_mace_predictions.pkl', args.model_name)
    eli5_annotation_df     = load_ds('../../data/eli5_ds/annotation-results/MACE-measure/final_mace_predictions.pkl', args.model_name)
    
    fivelvls_annotation_df['ds'] = ['5lvls'] * len(fivelvls_annotation_df)
    eli5_annotation_df['ds'] = ['eli5'] * len(eli5_annotation_df)
    dlgs_df = pd.concat([fivelvls_annotation_df, eli5_annotation_df])

    #split into train test split
    train_test_topics = {"train": {}, "test": {}}
    for dataset in ['eli5', '5lvls']:
        topics  = dlgs_df[dlgs_df.ds == dataset].topic.unique()
        train_topics, valid_topics = train_test_split(topics, shuffle=False, test_size=0.2, random_state=0)
        train_test_topics["train"][dataset] = list(train_topics)
        train_test_topics["test"][dataset] = list(valid_topics)
    
    label_clm = args.label_clm
    #for label_clm in ['dlg_act_label', 'exp_act_label', 'topic_func_label']:
    for ds in ['all']:
        if ds != 'all':
            working_dlgs_df = dlgs_df[dlgs_df.ds == ds].copy()
        else:
            working_dlgs_df = dlgs_df.copy()

        if len(working_dlgs_df) == 0:
            print('ds specified doesnt exist...')
            exit()

        output_path = '{}/{}/{}_prediction/{}_models/model/'.format(args.output_path, args.model_name, label_clm, ds)

        print('Training on {} with size {}'.format(ds, len(working_dlgs_df)))
        print('Output path {}'.format(output_path))
        
        if "deberta-v3" in args.model_name:
            batch_size = 4
        else:
            batch_size = 8
        
        eval_results = run_experiment(working_dlgs_df, train_test_topics, label_clm, args.input_clm, output_path, num_train_epochs=args.num_train_epochs, model_name=args.model_name, eval_steps=500, lr=2e-6, batch_size=batch_size)

        json.dump(eval_results, open('{}/eval_results.json'.format(output_path), 'w'))

        print(eval_results)