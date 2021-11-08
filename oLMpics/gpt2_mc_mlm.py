import argparse
from dataclasses import dataclass
import json
import logging
from os import system
import random
import numpy as np
import scipy.stats as st
import numpy as np
import pandas as pd
import torch 
from torch.utils.data import Dataset, DataLoader, SequentialSampler
import transformers
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s: %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

parser = argparse.ArgumentParser()
parser.add_argument("modelname", help="gpt2 model name")
parser.add_argument("results_seq_flag", help="True or False, set False for random division during evaluation ")


def get_data(file_path, sample, num_choices):
    """ Reads jsonl file (download links in readme)
    
    Arguments
        file_path (Jsonl file) : Path of the input file
        sample (Int) : -1 if samples needs to be randomly sampled
        num_choices () : 

    Returns:
        questions () : 
        choice_lists () :
        answer_ids () :

    """
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

def get_configuration():
    """ Set hyperparameters 
    Returns
    args () : Arguments values
    """
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
        num_choices=2,
    )
    return args

class BERTDataset(Dataset):
    """ 
    Data class for BERT and RoBERTa.
    """
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
    """ 
    Data class for RoBERTa.
    Only difference is that BERTDataset has token_type_ids while RoBERTaDataset doesn't
    """
    def __init__(self, questions, choices, answer_ids, tokenizer):
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
    """
    Computes the probability for a sentence in a batch

    Arguments:
        input_ids () :
        logits () :
        list_of_endtoken_index (list) : 

    Returns
        probs (FloatTensor) : 
    """
    # Multiplies together individual probabilities to get overall sentence probability
    logits = torch.nn.functional.softmax(logits, dim=2)
    probs = torch.gather(logits, 2, input_ids.unsqueeze(-1)[:, 1:]) # Shift the logit left by one for gpt2
    probs = probs.squeeze(-1)
    probs = probs * 1e4  # product is zero otherwise
    for i in range(len(list_of_endtoken_index)):     # Take probabilities till the tokens one before the end of token
      probs[i, list_of_endtoken_index[i]-1:] = 1
    probs = torch.sum(torch.log(probs), dim=1)
    return probs

def evaluate_mc_mlm(config, model, tokenizer, eval_dataset):
    """ 
    Evaluates model on the MC-MLM datasets 
    Arguments:
        args () :
        model () :
        tokenizer () :
        eval_dataset () :
        data_path () :

    Returns
        probs (FloatTensor) : 
    """
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=config.per_device_eval_batch_size)

    logger.info(f"***** Running evaluation  *****")
    logger.info(f"  Num examples = {len(eval_dataset)}")
    logger.info(f"  Batch size = {config.eval_batch_size}")
    eval_dataloader = tqdm(eval_dataloader, desc="Evaluating")
    
    #encoding all the labels
    label_dict = {}
    label_encodings = {}
    all_answers = []
    all_preds = []
    list_of_labels = eval_dataset[0]['choice_list']

    # Adding keys and values to dictionary label_encodings
    loop_counter=0
    for label in list_of_labels:
      label_dict[label] = loop_counter
      label1 = " "+ label
      label_encodings[label1] = tokenizer.encode(label1, add_special_tokens=False)[0]
      loop_counter+=1

    label_id_encoding_map = dict(zip(label_dict.values(),label_encodings.values()))
 
    for batch in eval_dataloader:       
        model.eval()
        # Creating the list all_answers  which has all the true labels 
        for loop_counter in range(len(batch["answer_id"])):
            true_label_id = batch["answer_id"][loop_counter]
            actual_label = batch["choice_list"][true_label_id][loop_counter]
            label_id_to_append = label_dict[actual_label]
            all_answers.append(label_id_to_append)
           
        del batch["choice_list"] 
        
        for key in batch:
            if key != "answer_id":
                batch[key] = torch.stack(batch[key], dim=-1)
            batch[key] = batch[key].cuda()
         
        _ = batch.pop("answer_id")
        label_encoding_list = list(label_encodings.values())
        no_of_labels = len(label_encoding_list)

        with torch.no_grad():           
            list_of_mask_index = []
            list_of_endtoken_index = []

            # Get sentence probabilities for a batch when [MASK] is replaced by label1
            for loop_counter in range(len(batch["input_ids"])):
                  question = batch["input_ids"][loop_counter]
                  MASK_INDEX = (question==tokenizer.mask_token_id).nonzero().item()
                  endtoken_index = (question==tokenizer.eos_token_id).nonzero()[0].item()
                  list_of_mask_index.append(MASK_INDEX)
                  list_of_endtoken_index.append(endtoken_index)

            id_prob = []
            
            for label_counter in range(no_of_labels):
                for loop_counter in range(len(batch["input_ids"])):
                    question = batch["input_ids"][loop_counter]
                    MASK_INDEX = list_of_mask_index[loop_counter]
                    batch["input_ids"][loop_counter, MASK_INDEX] =  label_id_encoding_map[label_counter]

                outputs = model(**batch)
                logits = outputs.logits
                batch_size = len(batch["input_ids"])
                id_prob[label_counter] = torch.reshape(get_sentence_prob(batch["input_ids"], logits, list_of_endtoken_index),  (batch_size, 1))

            combine_prob = torch.cat(tuple(id_prob), dim=1)
            preds = list(torch.argmax(combine_prob, dim=1))
            all_preds.extend(preds)
            
    return all_answers, all_preds

def zero_shot_evaluation_mc_mlm(config, dataset_dict, dataset_dict_seq,  model_name, results, results_seq, seq_flag):
    AgeDataset = RoBERTaDataset if any(prefix in model_name.lower() 
        for prefix in ("roberta", "bart", "distil", "gpt")) else BERTDataset

    model = transformers.AutoModelWithLMHead.from_pretrained(model_name).cuda()
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name , mask_token = '[MASK]')
    tokenizer.pad_token = tokenizer.eos_token # Each batch should have elements of same length and for gpt2 we need to define a pad token

    if seq_flag == 'False':
        print("Dividing the dataset RANDOMLY.")
        for task_name, num_choices in dataset_dict.items():
            accuracy = []
            for i in range(5):
              print("Evaluation {} with {} on {}".format(i, model_name, task_name))
              eval_questions, eval_choices, eval_answer_ids = get_data(task_name, config.sample_eval, num_choices)
              combined_dataset = {'que': eval_questions, 'choices': eval_choices, 'ids': eval_answer_ids, }
              combined_dataset = pd.DataFrame(data=combined_dataset)
              sampled_dataset = combined_dataset.sample(frac = 0.8)
              eval_questions = list(sampled_dataset['que'])
              eval_choices = list(sampled_dataset['choices'])
              eval_answer_ids = list(sampled_dataset['ids'])

              eval_dataset = AgeDataset(eval_questions, eval_choices, eval_answer_ids, tokenizer)
              all_answers, all_preds = evaluate_mc_mlm(config, model, tokenizer, eval_dataset, task_name)
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
    else:
        print("Dividing the dataset into five parts sequentially.")
        for task_name, num_choices in dataset_dict_seq.items():
              accuracy = []
              for i in range(5):
                  eval_questions, eval_choices, eval_answer_ids = get_data(task_name, config.sample_eval, num_choices)
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
                  all_answers, all_preds = evaluate_mc_mlm(config, model, tokenizer, eval_dataset, task_name)
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
              results_seq = results_seq.append(result_new, ignore_index=True)
        return results_seq

def main():
    config = get_configuration()
    args = parser.parse_args()
    transformers.set_seed(config.seed)

    dataset_dict = {"data/number_comparison_age_compare_masked_dev.jsonl":2, "data/negation_antonym_synonym_negation_dev.jsonl":2, "data/size_comparison_dev.jsonl":2, "data/compositional_comparison_dev.jsonl":3, "data/quantifiers_coffee_cats_quantifiers_dev.jsonl":5}
    dataset_dict_seq = {"data/number_comparison_age_compare_masked_dev.jsonl":2}
    model_name_or_path = args.modelname
    seq_flag = args.results_seq_flag

    results = pd.DataFrame(columns=["model_name", "task_name", "accuracy_5_runs", "accuracy_mean", "CI", "accuracy_min", "accuracy_max"])
    results_seq = pd.DataFrame(columns=["model_name", "task_name", "accuracy_5_runs", "accuracy_mean", "CI", "accuracy_min", "accuracy_max"])

    results = zero_shot_evaluation_mc_mlm(config, dataset_dict, dataset_dict_seq, model_name_or_path, results, results_seq, seq_flag)

    if seq_flag == 'False':
        if model_name_or_path == 'EleutherAI/gpt-neo-1.3B':
            results.to_excel('gpt2-results/gpt-neo-results.xlsx')
        elif  model_name_or_path == 'EleutherAI/gpt-j-6B':
            results.to_excel('gpt2-results/gpt-j-results.xlsx')
        else:
            results.to_excel('gpt2-results/{}-results.xlsx'.format(model_name_or_path))
    else:
        if model_name_or_path == 'EleutherAI/gpt-neo-1.3B':
            results.to_excel('gpt2-results/gpt-neo-seq-results.xlsx')
        elif  model_name_or_path == 'EleutherAI/gpt-j-6B':
            results.to_excel('gpt2-results/gpt-j-seq-results.xlsx')
        else:
            results.to_excel('gpt2-results/{}-seq-results.xlsx'.format(model_name_or_path))

if __name__ == '__main__':
    system.exit(main())  # next section explains the use of sys.exit