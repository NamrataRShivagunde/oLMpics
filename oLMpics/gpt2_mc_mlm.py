import argparse
from dataclasses import dataclass
import json
import logging
import os
import random
import sys
import time
import warnings
import random
import numpy as np
import scipy.stats as st
from pandas.io import parsers

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import transformers
from tqdm.auto import tqdm, trange

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s: %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

parser = argparse.ArgumentParser()
parser.add_argument("modelname", help = "gpt2 model name")
parser.add_argument("results_seq_flag", help = "True or False, set true for random division in evaluation ")


def get_data(file_path, sample, num_choices):
    """ Reads jsonl file (download links in readme) """
    data_file = open(file_path, "r")
    logger.info("Reading QA instances from jsonl dataset at: %s", file_path)
    item_jsons = []
    item_ids = []
    questions = []
    choice_lists = []
    answer_ids = []
    for line in data_file:
        item_jsons.append(json.loads(line.strip()))

    if sample != -1:
        item_jsons = random.sample(item_jsons, sample)
        logger.info("Sampling %d examples", sample)

    for item_json in tqdm(item_jsons,total=len(item_jsons)):
        item_id = item_json["id"]

        question_text = item_json["question"]["stem"]

        choice_label_to_id = {}
        choice_text_list = []
        choice_context_list = []
        choice_label_list = []
        choice_annotations_list = []

        any_correct = False
        choice_id_correction = 0

        for choice_id, choice_item in enumerate(item_json["question"]["choices"]):
            choice_label = choice_item["label"]
            choice_label_to_id[choice_label] = choice_id - choice_id_correction
            choice_text = choice_item["text"]

            choice_text_list.append(choice_text)
            choice_label_list.append(choice_label)

            if item_json.get('answerKey') == choice_label:
                if any_correct:
                    raise ValueError("More than one correct answer found for {item_json}!")
                any_correct = True

        if not any_correct and 'answerKey' in item_json:
            raise ValueError("No correct answer found for {item_json}!")

        answer_id = choice_label_to_id.get(item_json.get("answerKey"))
        # Pad choices with empty strings if not right number
        if len(choice_text_list) != num_choices:
            choice_text_list = (choice_text_list + num_choices * [''])[:num_choices]
            choice_context_list = (choice_context_list + num_choices * [None])[:num_choices]
            if answer_id is not None and answer_id >= num_choices:
                logging.warning(f"Skipping question with more than {num_choices} answers: {item_json}")
                continue

        item_ids.append(item_id)
        questions.append(question_text)
        choice_lists.append(choice_text_list)
        answer_ids.append(answer_id)

    data_file.close()
    return questions, choice_lists, answer_ids

@dataclass
class CustomArguments(transformers.TrainingArguments):
    sample_train: int = 0
    sample_eval: int = 0
    num_choices: int = 0
    model_name_or_path: str = "asdf"  # this is no longer a TrainingArgument attribute
        
    # python dataclasses cannot have positional attributes in subclass,
    # so give all attributes defaults and then make sure they are changed
    def __post_init__(self):
        if not (self.sample_train * self.sample_eval * self.num_choices) or \
               self.model_name_or_path == "asdf":  # make sure none are still default value
            raise TypeError("__init__ missing required argument(s)")

def get_args():
    """ Set hyperparameters """
    args = CustomArguments(
        output_dir="checkpoint",
        model_name_or_path="gpt2",
        overwrite_output_dir=True,
        do_train=False,  # Zero shot
        do_eval=True,
        per_device_eval_batch_size=2,
        learning_rate=1e-5,  # Should not matter because not training
        weight_decay=0.1,
        save_total_limit=2,
        seed=123,
        sample_train=200,
        sample_eval=-1,
        num_choices=2,
    )
    
    return args

class BERTDataset(Dataset):  # Only difference is that BERTDataset has token_type_ids while RoBERTaDataset doesn't
    
    def __init__(self, questions, choices, answer_ids, tokenizer):
        out = tokenizer(questions)
        self.input_ids = out["input_ids"]
        self.token_type_ids = out["token_type_ids"]
        self.attention_mask = out["attention_mask"]
        self.questions = questions
        self.choices = choices
        self.answer_ids = answer_ids
        
    def __len__(self):
        return len(self.questions)

    def __getitem__(self, i):
        return {
            "input_ids": self.input_ids[i], 
            "attention_mask": self.attention_mask[i], 
            "token_type_ids": self.token_type_ids[i],
            "choice_list": self.choices[i], 
            "answer_id": self.answer_ids[i],
        }
    

class RoBERTaDataset(Dataset):
    
    def __init__(self, questions, choices, answer_ids, tokenizer):
        if any(prefix in args.model_name_or_path.lower() for prefix in ("roberta", "bart")):
            questions = [question.replace('[MASK]','<mask>') for question in questions]
        out = tokenizer(questions, max_length=25, padding="max_length")
        self.input_ids = out["input_ids"]
        self.attention_mask = out["attention_mask"]
        self.questions = questions
        self.choices = choices
        self.answer_ids = answer_ids
        
    def __len__(self):
        return len(self.questions)

    def __getitem__(self, i):
        return {
            "input_ids": self.input_ids[i], 
            "attention_mask": self.attention_mask[i], 
            "choice_list": self.choices[i], 
            "answer_id": self.answer_ids[i],
        }

def get_sentence_prob(input_ids, logits, list_of_endtoken_index):
    # Multiplies together individual probabilities to get overall sentence probability
    logits = torch.nn.functional.softmax(logits, dim=2)
    probs = torch.gather(logits, 2, input_ids.unsqueeze(-1)[:, 1:]) # Shift the logit left by one for gpt2
    probs = probs.squeeze(-1)
    probs = probs * 1e4  # product is zero otherwise
    for i in range(len(list_of_endtoken_index)):     # Take probabilities till the tokens one before the end of token
      probs[i, list_of_endtoken_index[i]-1:] = 1
    probs = torch.sum(torch.log(probs), dim=1)
    return probs

def evaluate_mc_mlm(args, model, tokenizer, eval_dataset, data_path):
    """ 
    Evaluates gpt2 on the masked task dataset 
    """
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.per_device_eval_batch_size)

    logger.info(f"***** Running evaluation  *****")
    logger.info(f"  Num examples = {len(eval_dataset)}")
    logger.info(f"  Batch size = {args.eval_batch_size}")
    eval_dataloader = tqdm(eval_dataloader, desc="Evaluating")
    
    #encoding all the labels
    label_dict = {}
    label_encodings = {}
    list_of_labels = eval_dataset[0]['choice_list']
    i=0
    for label in list_of_labels:
      label_dict[label] = i
      label1 = " "+ label
      label_encodings[label1] = tokenizer.encode(label1, add_special_tokens=False)[0]
      i+=1

    label_id_encoding_map = dict(zip(label_dict.values(),label_encodings.values()))

    all_answers = []
    all_preds = []
    first_age = []
    second_age = []
    first_object = []
    second_object = []
    
    #create list of true answers =  all_answers 
    for batch in eval_dataloader:
        original_batch = batch          
        model.eval()
        for i in range(len(batch["answer_id"])):
            true_label_id = batch["answer_id"][i]
            actual_label = batch["choice_list"][true_label_id][i]
            label_id_to_append = label_dict[actual_label]
            all_answers.append(label_id_to_append)
           

        del batch["choice_list"] 
        for key in batch:
            if key != "answer_id":
                batch[key] = torch.stack(batch[key], dim=-1)

            batch[key] = batch[key].cuda()
      
        
        answer_ids = batch.pop("answer_id")
        label_encoding_list = list(label_encodings.values())
        no_of_labels = len(label_encoding_list)

        #create the list for age1, age2 for age task and list of objects for object comparison task
        if data_path == "data/number_comparison_age_compare_masked_dev.jsonl":
            age1 = tokenizer.decode(batch["input_ids"][:, 1]).split(" ")
            age2 = tokenizer.decode(batch["input_ids"][:, 11]).split(" ")
            age1 = age1[1:]
            age2 = age2[1:]
            first_age.extend(age1)
            second_age.extend(age2)


        with torch.no_grad():
            #generate probablities for all the labels
            
            list_of_mask_index = []
            list_of_endtoken_index = []
           
            for i in range(len(batch["input_ids"])):
                  question = batch["input_ids"][i]
                  MASK_INDEX = (question==tokenizer.mask_token_id).nonzero().item()
                  endtoken_index = (question==tokenizer.eos_token_id).nonzero()[0].item()
                  batch["input_ids"][i, MASK_INDEX] =  label_id_encoding_map[0]
                  list_of_mask_index.append(MASK_INDEX)
                  list_of_endtoken_index.append(endtoken_index)
            
            outputs = model(**batch)
            logits = outputs.logits
            id0_prob = get_sentence_prob(batch["input_ids"], logits, list_of_endtoken_index)
           
            m=0
            for i in range(len(batch["input_ids"])):
                question = batch["input_ids"][i]
                MASK_INDEX = list_of_mask_index[m]
                batch["input_ids"][i, MASK_INDEX] =  label_id_encoding_map[1]
                m +=1
      
            outputs = model(**batch)
            logits = outputs.logits
            id1_prob = get_sentence_prob(batch["input_ids"], logits, list_of_endtoken_index)
            
            if no_of_labels==3:
              m=0
              for i in range(len(batch["input_ids"])):
                  question = batch["input_ids"][i]
                  MASK_INDEX = list_of_mask_index[m]
                  batch["input_ids"][i, MASK_INDEX] =  label_id_encoding_map[2]
                  m +=1
                  
              outputs = model(**batch)
              logits = outputs.logits
              id2_prob = get_sentence_prob(batch["input_ids"], logits, list_of_endtoken_index)

            if no_of_labels==5:
                m=0
                for i in range(len(batch["input_ids"])):
                    question = batch["input_ids"][i]
                    MASK_INDEX = list_of_mask_index[m]
                    batch["input_ids"][i, MASK_INDEX] =  label_id_encoding_map[2]
                    m +=1
                
                outputs = model(**batch)
                logits = outputs.logits
                id2_prob = get_sentence_prob(batch["input_ids"], logits, list_of_endtoken_index)

                m=0
                for i in range(len(batch["input_ids"])):
                    question = batch["input_ids"][i]
                    MASK_INDEX = list_of_mask_index[m]
                    batch["input_ids"][i, MASK_INDEX] =  label_id_encoding_map[3]
                    m +=1
                outputs = model(**batch)
                logits = outputs.logits
                id3_prob = get_sentence_prob(batch["input_ids"], logits, list_of_endtoken_index)

                m=0
                for i in range(len(batch["input_ids"])):
                    question = batch["input_ids"][i]
                    MASK_INDEX = list_of_mask_index[m]
                    batch["input_ids"][i, MASK_INDEX] =  label_id_encoding_map[4]
                    m +=1
                  
                outputs = model(**batch)
                logits = outputs.logits
                id4_prob = get_sentence_prob(batch["input_ids"], logits, list_of_endtoken_index)
  
        batch_size = len(batch["input_ids"])
        #create all_preds
        if no_of_labels ==2:
          test_pred = torch.gt(id0_prob, id0_prob)
          id0_prob = torch.reshape(id0_prob, (batch_size, 1))
          id1_prob = torch.reshape(id1_prob, (batch_size, 1))
          combine_prob = torch.cat((id0_prob, id1_prob), dim=1)
          preds = list(torch.argmax(combine_prob, dim=1))
          all_preds.extend(preds)
        elif no_of_labels ==3:
          id0_prob = torch.reshape(id0_prob, (batch_size, 1))
          id1_prob = torch.reshape(id1_prob, (batch_size, 1))
          id2_prob = torch.reshape(id2_prob, (batch_size, 1))
          combine_prob = torch.cat((id0_prob, id1_prob, id2_prob), dim=1)
          preds = list(torch.argmax(combine_prob, dim=1))
          all_preds.extend(preds)
        elif no_of_labels ==5:
          id0_prob = torch.reshape(id0_prob, (batch_size, 1))
          id1_prob = torch.reshape(id1_prob,(batch_size, 1))
          id2_prob = torch.reshape(id2_prob, (batch_size, 1))
          id3_prob = torch.reshape(id3_prob, (batch_size, 1))
          id4_prob = torch.reshape(id4_prob, (batch_size, 1))
          combine_prob = torch.cat((id0_prob, id1_prob, id2_prob, id3_prob, id4_prob), dim=1)
          preds = list(torch.argmax(combine_prob, dim=1))
          all_preds.extend(preds)
    return all_answers, all_preds


args = get_args()
args2 = parser.parse_args()
transformers.set_seed(args.seed)

'''
models = "gpt" , "gpt-medium", "gpt2-large"
data  = "number_comparison_age_compare_masked_dev.jsonl" , args.num_choices = 2
data  = "antonym_synonym_negation_dev.jsonl" , args.num_choices = 2
data  = "size_comparison_dev.jsonl" , args.num_choices = 2
data  = "compositional_comparison_dev.jsonl" , args.num_choices = 3
data  = "coffee_cats_quantifiers_dev.jsonl" , args.num_choices = 5
'''
dataset_dict = {"data/number_comparison_age_compare_masked_dev.jsonl":2, "data/negation_antonym_synonym_negation_dev.jsonl":2, "data/size_comparison_dev.jsonl":2, "data/compositional_comparison_dev.jsonl":3, "data/quantifiers_coffee_cats_quantifiers_dev.jsonl":5}
dataset_dict_seq = {"data/number_comparison_age_compare_masked_dev.jsonl":2}
model_name_or_path = args2.modelname
seq_flag = args2.results_seq_flag

print(type(seq_flag))

model = transformers.AutoModelWithLMHead.from_pretrained(model_name_or_path).cuda()
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_or_path , mask_token = '[MASK]')
tokenizer.pad_token = tokenizer.eos_token

AgeDataset = RoBERTaDataset if any(prefix in model_name_or_path.lower() for prefix in ("roberta", "bart", "distil", "gpt")) else BERTDataset

results = pd.DataFrame(columns=["model_name", "task_name", "accuracy_5_runs", "accuracy_mean", "CI", "accuracy_min", "accuracy_max"])
results_seq = pd.DataFrame(columns=["model_name", "task_name", "accuracy_5_runs", "accuracy_mean", "CI", "accuracy_min", "accuracy_max"])


def zero_shot_evaluation_mc_mlm(dataset_dict, dataset_dict_seq,  model_name, results, results_seq, seq_flag):
    if seq_flag == 'False':
        print("Dividing the dataset RANDOMLY.")
        for task_name, num_choices in dataset_dict.items():
            accuracy = []
            for i in range(5):
              print("Evaluation {} with {} on {}".format(i, model_name, task_name))
              eval_questions, eval_choices, eval_answer_ids = get_data(task_name, args.sample_eval, num_choices)
              combined_dataset = {'que': eval_questions, 'choices': eval_choices, 'ids': eval_answer_ids, }
              combined_dataset = pd.DataFrame(data=combined_dataset)
              sampled_dataset = combined_dataset.sample(frac = 0.8)
              eval_questions = list(sampled_dataset['que'])
              eval_choices = list(sampled_dataset['choices'])
              eval_answer_ids = list(sampled_dataset['ids'])

              eval_dataset = AgeDataset(eval_questions, eval_choices, eval_answer_ids, tokenizer)
              all_answers, all_preds = evaluate_mc_mlm(args, model, tokenizer, eval_dataset, task_name)
              a = 0
              b = 0
              for i in range(len(all_answers)):
                if all_preds[i] != -1:
                    b += 1
                    if all_preds[i] == all_answers[i]:
                        a += 1
              current_acc = a/b
              accuracy.append(current_acc)
            mini, maxi = st.t.interval(alpha=0.95, df=len(accuracy)-1, loc=np.mean(accuracy), scale=st.sem(accuracy)) #sample size less than 30

            result_new = {'model_name': model_name, 'task_name':task_name, 'accuracy_5_runs':str(np.array(accuracy)), 'accuracy_mean': np.array(accuracy).mean()*100, 'CI':-1* np.array(accuracy).mean()*100+maxi*100, 'accuracy_min':mini, 'accuracy_max':maxi  }
            results = results.append(result_new, ignore_index=True)
        return results
    else:
        print("Dividing the dataset into five parts sequentially.")
        for task_name, num_choices in dataset_dict_seq.items():
              accuracy = []
              for i in range(5):
                  eval_questions, eval_choices, eval_answer_ids = get_data(task_name, args.sample_eval, num_choices)
                  total_items = len(eval_questions)
                  n = int(total_items/5)
                  if i==0:
                    eval_questions = eval_questions[n:]
                    eval_choices = eval_choices[n:]
                    eval_answer_ids = eval_answer_ids[n:] 
                  elif i==4:
                    eval_questions = eval_questions[:4*n]
                    eval_choices = eval_choices[:4*n]
                    eval_answer_ids = eval_answer_ids[:4*n]
                    
                  else:
                    eval_questions = eval_questions[:i*n] + eval_questions[(i+1)*n:]
                    eval_choices = eval_choices[:i*n] + eval_choices[(i+1)*n:]
                    eval_answer_ids = eval_answer_ids[:i*n] + eval_answer_ids[(i+1)*n:]   

                  eval_dataset = AgeDataset(eval_questions, eval_choices, eval_answer_ids, tokenizer)
                  all_answers, all_preds = evaluate_mc_mlm(args, model, tokenizer, eval_dataset, task_name)
                  a = 0
                  b = 0
                  for i in range(len(all_answers)):
                    if all_preds[i] != -1:
                        b += 1
                        if all_preds[i] == all_answers[i]:
                            a += 1
                  current_acc = a/b
                  accuracy.append(current_acc)
              
              mini, maxi = st.t.interval(alpha=0.95, df=len(accuracy)-1, loc=np.mean(accuracy), scale=st.sem(accuracy)) #sample size less than 30

              result_new = {'model_name': model_name, 'task_name':task_name, 'accuracy_5_runs':str(np.array(accuracy)), 'accuracy_mean': np.array(accuracy).mean()*100, 'CI':-1* np.array(accuracy).mean()*100+maxi*100, 'accuracy_min':mini, 'accuracy_max':maxi  }
              results_seq = results_seq.append(result_new, ignore_index=True)
        return results_seq

results = zero_shot_evaluation_mc_mlm(dataset_dict, dataset_dict_seq, model_name_or_path, results, results_seq, seq_flag)

if seq_flag == 'False':
    results.to_excel('gpt2-results/{}-results.xlsx'.format(model_name_or_path))
else:
    results.to_excel('gpt2-results/{}-seq-results.xlsx'.format(model_name_or_path))

