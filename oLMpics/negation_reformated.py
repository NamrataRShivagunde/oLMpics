import argparse
from dataclasses import dataclass
import json
import logging
import os
import random
import sys
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
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

def reformat_stem_pattern1(question_text):
  sent1, sent2 = question_text.split(',')
  sent11, sent12 = sent1.split('[MASK]')
  sent11= ' '.join(reversed(sent11.split(' '))).lower()
  reformatted_text = sent2.lstrip().capitalize() +" "+ sent11.lstrip().capitalize() + sent12 +' ?'
  return reformatted_text

def reformat_stem_pattern2_entails(question_text):
  sent1, sent2 = question_text.split(',')
  sent1 = sent1.replace('[MASK]','').lstrip()
  reformatted_text = sent1 + " entails"+sent2.lower().replace('.','?')
  return reformatted_text

def reformat_stem_pattern3(question_text):
  sent1, sent2 = question_text.split(',')
  sent11, sent12 = sent1.split('[MASK]')
  sent11= ' '.join(reversed(sent11.split(' '))).lower()
  reformatted_text = sent2.lstrip().capitalize() +" "+ sent11.lstrip().capitalize()  +" really" +sent12 + ' ?'
  return reformatted_text

def reformat_stem_pattern4(question_text):
  sent1, sent2 = question_text.split(',')
  sent1 = sent1.replace('[MASK]','').lstrip()
  reformatted_text = "Sentence 1: " + '"'+ sent1 + '.' +" Sentence 2: " + sent2 + " Is Sentence 1 synonym of Sentence 2 ?"
  return reformatted_text

def reformat_stem_pattern5(question_text):
  sent1, sent2 = question_text.split(',')
  sent1 = sent1.replace('[MASK]','').lstrip()
  reformatted_text = "Sentence1: " + '"'+ sent1 + '.' +" Sentence2: " + sent2 + " Is Sentence1 similar to Sentence2 ?"
  return reformatted_text

#test the reformatting
reformat_stem_pattern1('He was [MASK] ready to pillow, He was really ready to rest.')
#this should result in 'He was really ready to rest. Was he ready to pillow ?'
#He was really ready to rest. Was he ready to pillow ?

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
        

        #reformatting
        question_text = reformat_stem_pattern1(question_text)
       

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
        out = tokenizer(questions, max_length=40, padding="max_length")
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
def evaluate_qa_task(args, model, tokenizer, eval_dataset, data_path):
   
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.per_device_eval_batch_size)

    logger.info(f"***** Running evaluation  *****")
    logger.info(f"  Num examples = {len(eval_dataset)}")
    logger.info(f"  Batch size = {args.eval_batch_size}")
    eval_dataloader = tqdm(eval_dataloader, desc="Evaluating")

    all_answers = []
    all_preds = []


    for batch in eval_dataloader:
        model.eval()
        
        for i in range(len(batch["choice_list"][0])):
            all_answers.append(batch["choice_list"][batch["answer_id"][i]][i])

     
        choice_lists = batch.pop("choice_list")

        del batch["answer_id"] 
        for key in batch:
            batch[key] = torch.stack(batch[key], dim=-1).cuda()
      
        
        with torch.no_grad():       
          outputs = model(**batch)
          logits = outputs.logits
          logits = torch.nn.functional.softmax(logits, dim=2)
          #print(logits.size())
          choice_ids = []

          for i, logit in enumerate(logits):  
                first_pad_index = batch["input_ids"][i].tolist().index(tokenizer.eos_token_id)
                x =[" " + choice_lists[j][i] for j in range(len(choice_lists))]
                choice_ids = torch.tensor([tokenizer.encode(" " + choice_lists[j][i], add_special_tokens=False)[0] for j in range(len(choice_lists))])
              
                #hard coded for yes/no reformatting
                not_indices = choice_ids == 407
                really_indices = choice_ids == 1107
                choice_ids[not_indices] = 645
                choice_ids[really_indices] = 3763

                choice_ids = choice_ids.cuda()
                probs = logit[first_pad_index-1].index_select(0, choice_ids).cuda()
                max_ind = torch.argmax(probs)
                all_preds.append(choice_lists[max_ind][i])
            
    return all_answers, all_preds

args = get_args()
transformers.set_seed(args.seed)

'''
models = "gpt" , "gpt-medium", "gpt2-large", "gpt2-xl"
data  = "data/number_comparison_age_compare_masked_dev.jsonl" , args.num_choices = 2
data  = "data/negation_antonym_synonym_negation_dev.jsonl" , args.num_choices = 2
data  = "data/size_comparison_dev.jsonl" , args.num_choices = 2
data  = "data/compositional_comparison_dev.jsonl" , args.num_choices = 3
data  = "data/quantifiers_coffee_cats_quantifiers_dev.jsonl" , args.num_choices = 5
'''

args.num_choices = 2
args.model_name_or_path = 'gpt2'
data = "data/negation_antonym_synonym_negation_dev.jsonl"

#train_questions, train_choices, train_answer_ids = get_data(, args.sample_train, args.num_choices)
eval_questions, eval_choices, eval_answer_ids = get_data(data, args.sample_eval, args.num_choices)

model = transformers.AutoModelWithLMHead.from_pretrained(args.model_name_or_path).cuda()
tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name_or_path , mask_token = '[MASK]')
tokenizer.pad_token = tokenizer.eos_token

AgeDataset = RoBERTaDataset if any(prefix in args.model_name_or_path.lower() for prefix in ("roberta", "bart", "distil", "gpt")) else BERTDataset
#train_dataset = AgeDataset(train_questions, train_choices, train_answer_ids, tokenizer)
eval_dataset = AgeDataset(eval_questions, eval_choices, eval_answer_ids, tokenizer)

#eval_dataset[12]

all_answers, all_preds = evaluate_qa_task(args, model, tokenizer, eval_dataset, data)

tokenizer.convert_ids_to_tokens(50256)

#results
print("lenght of all_answers = ",len(all_answers))
print("lenght of all_preds = ",len(all_preds))
all_pred = []

total = 0
for i in range(len(all_answers)):
  if all_answers[i]==all_preds[i]:
    total +=1

accuracy = total/len(all_answers) 
print("The accuracy is of {} for {} task is ".format(args.model_name_or_path, data), accuracy)

