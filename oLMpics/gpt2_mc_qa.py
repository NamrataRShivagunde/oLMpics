import argparse
from dataclasses import dataclass
import json
import logging
import random
import numpy as np
import scipy.stats as st
import pandas as pd

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, SequentialSampler

import transformers
#import wandb
from tqdm.auto import tqdm


logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s: %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

parser = argparse.ArgumentParser()
parser.add_argument("modelname", help="gpt2 model name")

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

def get_configuration():
    """ Set hyperparameters """
    args = CustomArguments(
        output_dir="checkpoint",
        model_name_or_path="gpt2",
        overwrite_output_dir=True,
        do_train=False,  # Zero shot
        do_eval=True,
        per_device_eval_batch_size=8,
        learning_rate=1e-5,  # Should not matter because not training
        weight_decay=0.1,
        save_total_limit=2,
        seed=123,
        sample_train=200,
        sample_eval=-1,
        num_choices=3,
    )
    
    return args

def get_data(file_path, sample, num_choices):
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
        out = tokenizer(questions, max_length=45, padding="max_length")
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

def evaluate_qa_task(config, model, tokenizer, eval_dataset, data_path):
   
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=config.per_device_eval_batch_size)

    logger.info(f"***** Running evaluation  *****")
    logger.info(f"  Num examples = {len(eval_dataset)}")
    logger.info(f"  Batch size = {config.eval_batch_size}")
    eval_dataloader = tqdm(eval_dataloader, desc="Evaluating")

    all_answers = []
    all_preds = []

    MASK_ID = tokenizer.encode(tokenizer.mask_token)
    MASK_ID = MASK_ID[0]

    for batch in eval_dataloader:
        model.eval()
        
        for i in range(len(batch["choice_list"][0])):
            all_answers.append(batch["choice_list"][batch["answer_id"][i]][i])
     
        choice_lists = batch.pop("choice_list")

        del batch["answer_id"] 
        for key in batch:
            batch[key] = torch.stack(batch[key], dim=-1).cuda()
      
        
        with torch.no_grad():
          if data_path == "hypernym_conjunction_dev.jsonl":
            #replace [MASK] with the index of first pad token as it will be the last token 
            for i in range(len(batch["input_ids"])):
                  question = batch["input_ids"][i]
                  MASK_INDEX = (question==tokenizer.mask_token_id).nonzero().item()
                  batch["input_ids"][i, MASK_INDEX] = 220
            
          outputs = model(**batch)
          logits = outputs.logits
          logits = torch.nn.functional.softmax(logits, dim=2)
          choice_ids = []

          for i, logit in enumerate(logits):  
                first_pad_index = batch["input_ids"][i].tolist().index(tokenizer.eos_token_id)
                x =[" " + choice_lists[j][i] for j in range(len(choice_lists))]
                choice_ids = torch.tensor([tokenizer.encode(" " + choice_lists[j][i], add_special_tokens=False)[0] for j in range(len(choice_lists))])
                choice_ids = choice_ids.cuda()
                probs = logit[first_pad_index-1].index_select(0, choice_ids).cuda()
                max_ind = torch.argmax(probs)
                all_preds.append(choice_lists[max_ind][i])
 
    return all_answers, all_preds

def zero_shot_evaluation(config, dataset_dict, model_name, results):
    #for loop for each task
    AgeDataset = RoBERTaDataset if any(prefix in model_name.lower() for prefix in ("roberta", "bart", "distil", "gpt")) else BERTDataset
    
    model = transformers.AutoModelWithLMHead.from_pretrained(model_name).cuda()
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name , mask_token = '[MASK]')
    tokenizer.pad_token = tokenizer.eos_token # Each batch should have elements of same length and for gpt2 we need to define a pad token
    
    for task_name, num_choices in dataset_dict.items():
            accuracy = []
            for i in range(5):
                print("Evaluation {} with {} on {}".format(i, model_name, task_name))
                eval_questions, eval_choices, eval_answer_ids = get_data(task_name, config.sample_eval, num_choices)
                combined_dataset = {'que': eval_questions, 'choices': eval_choices, 'ids': eval_answer_ids }
                combined_dataset = pd.DataFrame(data=combined_dataset)
                sampled_dataset = combined_dataset.sample(frac = 0.8)
                eval_questions = list(sampled_dataset['que'])
                eval_choices = list(sampled_dataset['choices'])
                eval_answer_ids = list(sampled_dataset['ids'])

                eval_dataset = AgeDataset(eval_questions, eval_choices, eval_answer_ids, tokenizer)
                all_answers, all_preds = evaluate_qa_task(config, model, tokenizer, eval_dataset, task_name)
                counter_a = 0
                counter_b = 0
                for i in range(len(all_answers)):
                    if all_preds[i] != -1:
                        counter_b += 1
                        if all_preds[i] == all_answers[i]:
                            counter_a += 1
                current_acc = counter_a/counter_b
                accuracy.append(current_acc)

            mini, maxi = st.t.interval(alpha=0.95, df=len(accuracy)-1, loc=np.mean(accuracy), scale=st.sem(accuracy)) #sample size less than 30

            result_new = {'model_name': model_name, 'task_name':task_name, 'accuracy_5_runs':str(np.array(accuracy)), 'accuracy_mean': np.array(accuracy).mean()*100, 'CI':-1* np.array(accuracy).mean()*100+maxi*100, 'accuracy_min':mini, 'accuracy_max':maxi  }
            results = results.append(result_new, ignore_index=True)
    return results

def main():
        args = parser.parse_args()
        config = get_configuration()
        transformers.set_seed(config.seed)

        dataset_dict = {"data/hypernym_conjunction_dev.jsonl":3, "data/composition_composition_v2_dev.jsonl":3, "data/conjunction_conjunction_filt4_dev.jsonl":3}
        results = pd.DataFrame(columns=["model_name", "task_name", "accuracy_5_runs", "accuracy_mean", "CI", "accuracy_min", "accuracy_max"])

        results = zero_shot_evaluation(config, dataset_dict, args.modelname, results)

        if args.modelname == 'EleutherAI/gpt-neo-1.3B':
            results.to_excel('gpt2-results/gpt-neo-results.xlsx')
        elif  args.modelname == 'EleutherAI/gpt-j-6B':
            results.to_excel('gpt2-results/gpt-j-results.xlsx')
        else:
            results.to_excel('gpt2-results/{}-results.xlsx'.format(args.model_name_or_path))
        

if __name__ == '__main__':
    main()