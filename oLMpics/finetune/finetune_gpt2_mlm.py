import argparse
from dataclasses import dataclass
import json
import logging
from os import system
import random
import wandb
import numpy as np
import scipy.stats as st
import numpy as np
import pandas as pd
import torch 
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import transformers
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s: %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


def get_args():
    """ Set hyperparameters """
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name_or_path",
    help="Huggingface pretrained model name/path")

    parser.add_argument("train_data_path",
    help="Path to jsonl training data for MLM task")

    parser.add_argument("eval_data_path",
    help="Path to jsonl development data for MLM task")

    parser.add_argument("num_choices", type=int,
    help="Number of answer choices for task")

    parser.add_argument(
        "--max_seq_length",
        default=25,
        type=int,
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        default=32,
        type=int,
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        default=64,
        type=int,
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        default=1,
        type=int,
    )
    parser.add_argument(
        "--num_train_epochs",
        default=4,
        type=int,
    )
    parser.add_argument(
        "--learning_rate",
        default=5e-5,
        type=float,
    )
    parser.add_argument(
        "--weight_decay",
        default=0.1,
        type=float,
    )
    parser.add_argument(
        "--no_dropout",
        action="store_true"
    )
    parser.add_argument(
        "--warmup_ratio",
        default=0.06,
        type=float,
    )
    parser.add_argument(
        "--seed",
        default=123,
        type=int,
    )
    parser.add_argument(
        "--sample_train",
        default=-1,
        type=int,
        help="Number of examples (not batches) to evaluate on, \
        default of -1 evaluates on entire dataset"
    )
    parser.add_argument(
        "--sample_eval",
        default=-1,
        type=int,
        help="Number of examples (not batches) to evaluate on, \
        default of -1 evaluates on entire dataset"
    )
    parser.add_argument(
        "--clip",
        default=1,
        type=float,
        help="Argument for gradient clipping, -1 means no clipping"
    )
    parser.add_argument(
        "--device",
        default="cpu"
        #default="cuda" if torch.cuda.is_available() else "cpu"
    )

    args = parser.parse_args()
    return args

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

class BERTDataset(Dataset):
    """ Dataset with token_type_ids (used for BERT, ALBERT) """
    def __init__(self, questions, choices, answer_ids, tokenizer, max_length):
        out = tokenizer(questions, max_length=max_length, padding="max_length", truncation=True)
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
    """ Dataset without token_type_ids (used for RoBERTa, BART, Distil, ELECTRA, T5) """
    def __init__(self, questions, choices, answer_ids, tokenizer, max_length):
        questions = [question.replace('[MASK]', tokenizer.mask_token) for question in questions]
        out = tokenizer(questions, max_length=max_length, padding="max_length", truncation=True)
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


def evaluate(args, model, tokenizer, eval_dataset, is_train=False):
    """
    Args:
        args:
            hyperparameters set using get_args()
        model:
            Huggingface model which will be used for evaluation
        tokenizer:
            Huggingface tokenizer

    Returns: Tuple of (answers, preds)
        answers - list of ground-truth labels
        preds - list of labels predicted by model
    """
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.per_device_eval_batch_size)

    logger.info(f"***** Running evaluation  *****")
    logger.info(f"  Num examples = {len(eval_dataset)}")
    logger.info(f"  Batch size = {args.per_device_eval_batch_size}")
    eval_dataloader = tqdm(eval_dataloader, desc="Evaluating")
    
    #encoding all the labels
    label_dict = {}
    label_encodings = {}
    all_answers = []
    all_preds = []
    eval_loss = 0
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
        original_batch = batch
        labels_for_eval_loss = []
        # Creating the list all_answers  which has all the true labels 
        for loop_counter in range(len(batch["answer_id"])):
            true_label_id = batch["answer_id"][loop_counter]
            actual_label = batch["choice_list"][true_label_id][loop_counter]
            label_id_to_append = label_dict[actual_label]
            all_answers.append(label_id_to_append)
            labels_for_eval_loss.append(actual_label)
           
        del batch["choice_list"] 
        
        for key in batch:
            if key != "answer_id":
                batch[key] = torch.stack(batch[key], dim=-1)
            batch[key] = batch[key].to(args.device)
         
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

                outputs_prob = model(**batch)
                logits = outputs_prob.logits

                batch_size = len(batch["input_ids"])
                id_prob.append(
                    torch.reshape(get_sentence_prob(batch["input_ids"], logits, list_of_endtoken_index),  
                    (batch_size, 1)))

            combine_prob = torch.cat(tuple(id_prob), dim=1)
            preds = list(torch.argmax(combine_prob, dim=1))
            all_preds.extend(preds)

            # to get eval_loss, create labels and pass it as arguments to model
            for i in range(len(batch["input_ids"])):
                question = batch["input_ids"][i]
                MASK_INDEX = list_of_mask_index[i]
                batch["input_ids"][i, MASK_INDEX] =  labels_for_eval_loss[i]
            
            outputs = model(**original_batch, label =  batch["input_ids"])    
            eval_loss += outputs.loss

    eval_loss /= len(eval_dataset)
    if is_train:
        wandb.log({"avg_train_loss": eval_loss})
    else:
        wandb.log({"avg_eval_loss": eval_loss})

    return (np.array(all_answers) == np.array(all_preds)).mean()

def train(args, model, tokenizer, train_dataset, eval_dataset):
    eval_acc = evaluate(args, model, tokenizer, eval_dataset)
    logger.info(f"Initial Eval Accuracy: {eval_acc}")
    train_acc = evaluate(args, model, tokenizer, train_dataset, is_train=True)
    logger.info(f"Initial Train Accuracy: {train_acc}")
    wandb.log({"eval_acc": eval_acc, "train_acc": train_acc})

    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.per_device_train_batch_size)

    logger.info(f"***** Running training  *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Batch size = {args.per_device_train_batch_size}")
    train_dataloader = tqdm(train_dataloader, desc="Training", leave=False)

    optimizer = transformers.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = transformers.get_linear_schedule_with_warmup(optimizer, len(train_dataloader) * args.num_train_epochs // args.gradient_accumulation_steps * args.warmup_ratio, len(train_dataloader) * args.num_train_epochs // args.gradient_accumulation_steps)

    MASK_ID = tokenizer.encode(tokenizer.mask_token, add_special_tokens=False)
    assert len(MASK_ID) == 1
    MASK_ID = MASK_ID[0]
    accumulated_loss = torch.tensor(0.0).to(args.device)

    for epoch in tqdm(range(args.num_train_epochs)):
        for step, batch in enumerate(train_dataloader):
            if args.no_dropout:
                model.eval()
            else:
                model.train()
            
            labels = []
            # batch["choice_list"] is [num_choices, batch_size]
            for loop_counter in range(len(batch["answer_id"])):
                true_label_id = batch["answer_id"][loop_counter]
                actual_label = batch["choice_list"][true_label_id][loop_counter]
                labels.append(actual_label)

            # to get eval_loss, create labels and pass it as arguments to model
            for i in range(len(batch["input_ids")):
                question = batch["input_ids"][i]
                MASK_INDEX = (question==tokenizer.mask_token_id).nonzero().item()
                batch["input_ids"][i, MASK_INDEX] =  labels[i]
            
            outputs = model(**original_batch, label =  batch["input_ids"])    
            loss = outputs.loss
            
            accumulated_loss += float(loss)
            if (step + 1) % args.gradient_accumulation_steps == 0:
                wandb.log({"train_loss": accumulated_loss.item()})
                wandb.log({"lr": scheduler.get_last_lr()[0]})
                loss.backward()
                if args.clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

                optimizer.step()
                scheduler.step()
                model.zero_grad()
                accumulated_loss = torch.tensor(0.0).to(args.device)

        eval_acc = evaluate(args, model, tokenizer, eval_dataset)
        logger.info(f"{epoch}th Eval Accuracy: {eval_acc}")
        train_acc = evaluate(args, model, tokenizer, train_dataset, is_train=True)
        logger.info(f"{epoch}th Train Accuracy: {train_acc}")
        wandb.log({"eval_acc": eval_acc, "train_acc": train_acc})

    return True

def main():
    args = get_args()

    wandb.init(project="oLMpics", entity="namratashivagunde", group="Nov1", \
               name=f"{args.model_name_or_path}_{args.train_data_path[5:-6]}")
    wandb.config.update(args)

    transformers.set_seed(args.seed)

    logger.info("Loading model.")

    transformers.set_seed(args.seed)

    model = transformers.AutoModelWithLMHead.from_pretrained(args.model_name_or_path).to(args.device)
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name_or_path , mask_token = '[MASK]')
    tokenizer.pad_token = tokenizer.eos_token # Each batch should have elements of same length and for gpt2 we need to define a pad token
    model.resize_token_embeddings(len(tokenizer))

    AgeDataset = RoBERTaDataset if any(prefix in args.model_name_or_path.lower() \
    for prefix in ("roberta", "bart", "distil", "electra", "t5", "gpt")) else BERTDataset

    wandb.watch(model, log_freq=50)
    train_questions, train_choices, train_answer_ids = get_data(args.train_data_path, args.sample_train, args.num_choices)
    eval_questions, eval_choices, eval_answer_ids = get_data(args.eval_data_path, args.sample_eval, args.num_choices)

    train_dataset = AgeDataset(train_questions, train_choices, train_answer_ids, tokenizer, args.max_seq_length)
    eval_dataset = AgeDataset(eval_questions, eval_choices, eval_answer_ids, tokenizer, args.max_seq_length)

    train(args, model, tokenizer, train_dataset, eval_dataset)

if __name__ == '__main__':
    main()
