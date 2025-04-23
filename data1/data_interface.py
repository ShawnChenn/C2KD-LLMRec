import inspect
import importlib
import pickle as pkl
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
import random
import torch
import argparse
from transformers import LlamaForCausalLM, LlamaTokenizer
import os
import numpy as np
from collections import defaultdict
from torch.nn.utils.rnn import pad_sequence
from torch.nn import functional as F

class TrainCollater:
    def __init__(self,
                 prompt_list=None,
                 llm_tokenizer=None,
                 train=False,
                 terminator="\n",
                 max_step=1):
        self.prompt_list = prompt_list
        self.llm_tokenizer = llm_tokenizer
        self.train=train
        self.terminator = terminator
        self.max_step = max_step
        self.cur_step = 1

    def __call__(self, batch):
        if isinstance(self.prompt_list,list):
            instruction = random.choice(self.prompt_list)
            inputs_text = instruction if isinstance(instruction, list) else [instruction] * len(batch)
        else:
            instruction = sample["instruction_input"] if "instruction_input" in sample else None
            inputs_text = instruction if isinstance(instruction, list) else [instruction] * len(batch)
        
        thresh_hold = self.cur_step/self.max_step
        p = random.random()
        if p < thresh_hold or not self.train:
            for i, sample in enumerate(batch):
                input_text=inputs_text[i]
                if '[HistoryHere]' in input_text:
                    insert_prompt=", ".join([seq_title+' [HistoryEmb]' for seq_title in sample['seq_name']])
                    input_text=input_text.replace('[HistoryHere]',insert_prompt)
                if '[CansHere]' in input_text:
                    insert_prompt=", ".join([can_title+' [CansEmb]' for can_title in sample['cans_name']])
                    input_text=input_text.replace('[CansHere]',insert_prompt)    
                inputs_text[i]=input_text
            flag = False
        else:
            for i, sample in enumerate(batch):
                input_text=inputs_text[i]
                if '[HistoryHere]' in input_text:
                    insert_prompt=", ".join([seq_title+' [PH]' for seq_title in sample['seq_name']])
                    input_text=input_text.replace('[HistoryHere]',insert_prompt)
                if '[CansHere]' in input_text:
                    insert_prompt=", ".join([can_title+' [PH]' for can_title in sample['cans_name']])
                    input_text=input_text.replace('[CansHere]',insert_prompt)    
                inputs_text[i]=input_text
            flag = True
        self.cur_step += 1
        
        targets_text = [sample['correct_answer'] for sample in batch]
        cans_text = [sample['cans_name'] for sample in batch]

        if self.train:
            targets_text=[target_text+self.terminator for target_text in targets_text]
            inputs_pair = [[p, t] for p, t in zip(inputs_text, targets_text)]

            batch_tokens = self.llm_tokenizer(
                inputs_pair,
                return_tensors="pt",
                padding="longest",
                truncation=False,
                add_special_tokens=True,
                return_attention_mask=True,
                return_token_type_ids=True)
           
            new_batch={"teacher_tokens":batch_tokens,
                       "seq":torch.stack([torch.tensor(sample['seq']) for sample in batch], dim=0),
                       "cans":torch.stack([torch.tensor(sample['cans']) for sample in batch], dim=0),
                       "len_seq":torch.stack([torch.tensor(sample['len_seq']) for sample in batch], dim=0),
                       "len_cans":torch.stack([torch.tensor(sample['len_cans']) for sample in batch], dim=0),
                       "item_id": torch.stack([torch.tensor(sample['item_id']) for sample in batch], dim=0),
                       "flag":flag,
                       }
        else:
            batch_tokens = self.llm_tokenizer(
                inputs_text,
                return_tensors="pt",
                padding="longest",
                truncation=False,
                add_special_tokens=True,
                return_attention_mask=True)
            cans_name=[sample['cans_name'] for sample in batch]
            new_batch={"teacher_tokens":batch_tokens,
                       "seq":torch.stack([torch.tensor(sample['seq']) for sample in batch], dim=0),
                       "cans":torch.stack([torch.tensor(sample['cans']) for sample in batch], dim=0),
                       "len_seq":torch.stack([torch.tensor(sample['len_seq']) for sample in batch], dim=0),
                       "len_cans":torch.stack([torch.tensor(sample['len_cans']) for sample in batch], dim=0),
                       "item_id": torch.stack([torch.tensor(sample['item_id']) for sample in batch], dim=0),
                       "correct_answer": targets_text,
                       "cans_name": cans_name,
                       }
        return new_batch

class Distill_TrainCollater:
    def __init__(self,
                 prompt_list=None,
                 teacher_llm_tokenizer=None,
                 student_llm_tokenizer=None,
                 train=False,
                 terminator="\n",
                 max_step=1):
        self.prompt_list = prompt_list
        self.teacher_llm_tokenizer = teacher_llm_tokenizer
        self.student_llm_tokenizer = student_llm_tokenizer
        self.train=train
        self.terminator = terminator
        self.max_step = max_step
        self.cur_step = 1
        self.label_pad_token_id = -100
    
    def build_tokenized_answer(self, tokenizer, prompt, answer):
        """
        Llama tokenizer does satisfy `enc(a + b) = enc(a) + enc(b)`.
        It does ensure `enc(a + b) = enc(a) + enc(a + b)[len(enc(a)):]`.
        Reference:
            https://github.com/EleutherAI/lm-evaluation-harness/pull/531#issuecomment-1595586257
        """
        full_tokenized = tokenizer(prompt + answer, add_special_tokens=False)
        prompt_input_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]

        answer_input_ids = full_tokenized["input_ids"][len(prompt_input_ids) :]
        answer_attention_mask = full_tokenized["attention_mask"][len(prompt_input_ids) :]

        # Concat tokens to form `enc(a) + enc(a + b)[len(enc(a)):]`
        full_concat_input_ids = np.concatenate([prompt_input_ids, answer_input_ids])

        # Prepare input tokens for token by token comparison
        full_input_ids = np.array(full_tokenized["input_ids"])

        if len(full_input_ids) != len(full_concat_input_ids):
            raise ValueError("Prompt input ids and answer input ids should have the same length.")

        # On some tokenizers, like Llama-2 tokenizer, there are occasions where tokens
        # can be merged together when tokenizing prompt+answer. This could result
        # on the last token from the prompt being different when tokenized on its own
        # vs when done as prompt+answer.
        response_token_ids_start_idx = len(prompt_input_ids)

        # If tokenized prompt is different than both prompt+answer, then it means the
        # last token has changed due to merging.
        if prompt_input_ids != full_tokenized["input_ids"][:response_token_ids_start_idx]:
            response_token_ids_start_idx -= 1

        prompt_input_ids = full_tokenized["input_ids"][:response_token_ids_start_idx]
        prompt_attention_mask = full_tokenized["attention_mask"][:response_token_ids_start_idx]

        if len(prompt_input_ids) != len(prompt_attention_mask):
            raise ValueError("Prompt input ids and attention mask should have the same length.")

        answer_input_ids = full_tokenized["input_ids"][response_token_ids_start_idx:]
        answer_attention_mask = full_tokenized["attention_mask"][response_token_ids_start_idx:]

        return dict(
            prompt_input_ids=prompt_input_ids,
            prompt_attention_mask=prompt_attention_mask,
            input_ids=answer_input_ids,
            attention_mask=answer_attention_mask,
        )

    def tokenize_row(self, prompt, chosen, reject, tokenizer):
        batch = {}
        prompt_tokens = tokenizer(prompt, add_special_tokens=False)
        prompt_tokens = {f"prompt_{k}": v for k, v in prompt_tokens.items()}
        # 'prompt_teacher_input_ids', 'prompt_teacher_attention_mask'
            
        chosen_tokens = self.build_tokenized_answer(tokenizer, prompt, chosen)
        rejected_tokens = self.build_tokenized_answer(tokenizer, prompt, reject)
            
        # Last prompt token might get merged by tokenizer and
        # it should not be included for generation if that happens
        prompt_len_input_ids = len(prompt_tokens["prompt_input_ids"])
        chosen_prompt_len_input_ids = len(chosen_tokens["prompt_input_ids"])
        rejected_prompt_len_input_ids = len(rejected_tokens["prompt_input_ids"])
        prompt_len_input_ids = min(chosen_prompt_len_input_ids, rejected_prompt_len_input_ids)

        for k, v in prompt_tokens.items():
            prompt_tokens[k] = v[:prompt_len_input_ids]

        # Make sure prompts only have one different token at most an
        # and length only differs by 1 at most
        num_diff_tokens = sum([a != b for a, b in zip(chosen_tokens["prompt_input_ids"], rejected_tokens["prompt_input_ids"])])
        num_diff_len = abs(chosen_prompt_len_input_ids - rejected_prompt_len_input_ids)
        if num_diff_tokens > 1 or num_diff_len > 1:
            raise ValueError(
                    "Chosen and rejected prompt_input_ids might only differ on the "
                    "last token due to tokenizer merge ops.")

        # add BOS token to head of prompt. Avoid adding if it's already there
        bos_token_id = tokenizer.bos_token_id
        if prompt_len_input_ids == 0 or bos_token_id != prompt_tokens["prompt_input_ids"][0]:
            prompt_tokens["prompt_input_ids"] = [bos_token_id] + prompt_tokens["prompt_input_ids"]
            prompt_tokens["prompt_attention_mask"] = [1] + prompt_tokens["prompt_attention_mask"]
        if chosen_prompt_len_input_ids == 0 or bos_token_id != chosen_tokens["prompt_input_ids"][0]:
            chosen_tokens["prompt_input_ids"] = [bos_token_id] + chosen_tokens["prompt_input_ids"]
            chosen_tokens["prompt_attention_mask"] = [1] + chosen_tokens["prompt_attention_mask"]
        if rejected_prompt_len_input_ids == 0 or bos_token_id != rejected_tokens["prompt_input_ids"][0]:
            rejected_tokens["prompt_input_ids"] = [bos_token_id] + rejected_tokens["prompt_input_ids"]
            rejected_tokens["prompt_attention_mask"] = [1] + rejected_tokens["prompt_attention_mask"]

        # add EOS token to end of answer. Avoid adding if it's already there
        eos_token_id = tokenizer.eos_token_id
        if len(chosen_tokens["input_ids"]) == 0 or eos_token_id != chosen_tokens["input_ids"][-1]:
            chosen_tokens["input_ids"].append(eos_token_id)
            chosen_tokens["attention_mask"].append(1)
        if len(rejected_tokens["input_ids"]) == 0 or eos_token_id != rejected_tokens["input_ids"][-1]:
            rejected_tokens["input_ids"].append(eos_token_id)
            rejected_tokens["attention_mask"].append(1)

        # longer_response_length = max(len(chosen_tokens["input_ids"]), len(rejected_tokens["input_ids"]))

        # Create labels
        chosen_sequence_tokens = {
                k: chosen_tokens[f"prompt_{k}"] + chosen_tokens[k] for k in ["input_ids", "attention_mask"]
        }
        rejected_sequence_tokens = {
                k: rejected_tokens[f"prompt_{k}"] + rejected_tokens[k] for k in ["input_ids", "attention_mask"]
        }
        chosen_sequence_tokens["labels"] = chosen_sequence_tokens["input_ids"][:]
        chosen_sequence_tokens["labels"][: len(chosen_tokens["prompt_input_ids"])] = [
                self.label_pad_token_id
            ] * len(chosen_tokens["prompt_input_ids"])
        rejected_sequence_tokens["labels"] = rejected_sequence_tokens["input_ids"][:]
        rejected_sequence_tokens["labels"][: len(rejected_tokens["prompt_input_ids"])] = [
                self.label_pad_token_id
            ] * len(rejected_tokens["prompt_input_ids"])

        for k, toks in {"chosen_": chosen_sequence_tokens, "rejected_": rejected_sequence_tokens, "": prompt_tokens,}.items():
            for type_key, tokens in toks.items():
                if type_key == "token_type_ids":
                    continue
                batch[f"{k}{type_key}"] = tokens
        return batch
        
    def __call__(self, batch):
        if isinstance(self.prompt_list,list):
            instruction = random.choice(self.prompt_list)
            inputs_text = instruction if isinstance(instruction, list) else [instruction] * len(batch)
        else:
            instruction = sample["instruction_input"] if "instruction_input" in sample else None
            inputs_text = instruction if isinstance(instruction, list) else [instruction] * len(batch)
        
        thresh_hold = self.cur_step/self.max_step
        p = random.random()
        if p < thresh_hold or not self.train:
            for i, sample in enumerate(batch):
                input_text=inputs_text[i]
                if '[HistoryHere]' in input_text:
                    insert_prompt=", ".join([seq_title+' [HistoryEmb]' for seq_title in sample['seq_name']])
                    input_text=input_text.replace('[HistoryHere]',insert_prompt)
                if '[CansHere]' in input_text:
                    insert_prompt=", ".join([can_title+' [CansEmb]' for can_title in sample['cans_name']])
                    input_text=input_text.replace('[CansHere]',insert_prompt)    
                inputs_text[i]=input_text
            flag = False
        else:
            for i, sample in enumerate(batch):
                input_text=inputs_text[i]
                if '[HistoryHere]' in input_text:
                    insert_prompt=", ".join([seq_title+' [PH]' for seq_title in sample['seq_name']])
                    input_text=input_text.replace('[HistoryHere]',insert_prompt)
                if '[CansHere]' in input_text:
                    insert_prompt=", ".join([can_title+' [PH]' for can_title in sample['cans_name']])
                    input_text=input_text.replace('[CansHere]',insert_prompt)    
                inputs_text[i]=input_text
            flag = True
        self.cur_step += 1
        
        targets_text = [sample['correct_answer'] for sample in batch]
        cans_text = [sample['cans_name'] for sample in batch]

        if self.train:
            targets_text=[target_text + self.terminator for target_text in targets_text]
            inputs_pair = [[p, t] for p, t in zip(inputs_text, targets_text)]
            
            combined_teacher_batch ={}
            combined_student_batch = {}

            for prompt, chosen, cans in zip(inputs_text, targets_text, cans_text):
                remaining_candidates = [cand for cand in cans if cand != chosen]
                rej = random.sample(remaining_candidates, 1)[0]
             
                batch_row = self.tokenize_row(prompt, chosen, rej, self.teacher_llm_tokenizer)
                for key, value in batch_row.items():
                    if key not in combined_teacher_batch:
                        combined_teacher_batch[key] = []
                    if key not in combined_student_batch:
                        combined_student_batch[key] = []
                    combined_teacher_batch[key].append(value)
                    combined_student_batch[key].append(value)
            
            batch_teacher_tokens = self.teacher_llm_tokenizer(
                inputs_pair,
                return_tensors="pt",
                padding="longest",
                truncation=False,
                add_special_tokens=True,
                return_attention_mask=True,
                return_token_type_ids=True)
            
            batch_student_tokens = self.student_llm_tokenizer(
                inputs_pair,
                return_tensors="pt",
                padding="longest",
                truncation=False,
                add_special_tokens=True,
                return_attention_mask=True,
                return_token_type_ids=True)
            
            for k in combined_teacher_batch.keys():               
                tensors = [torch.tensor(lst) for lst in combined_teacher_batch[k]] 
                if 'label' in k:
                    batch_tensor = pad_sequence(tensors, batch_first=True, padding_value=-100)
                    combined_teacher_batch[k] = batch_tensor
                else:
                    batch_tensor = pad_sequence(tensors, batch_first=True, padding_value=0)                
                    combined_teacher_batch[k] = batch_tensor
            
            for k in combined_student_batch.keys():
                tensors = [torch.tensor(lst) for lst in combined_student_batch[k]] 
                if 'label' in k:
                    batch_tensor = pad_sequence(tensors, batch_first=True, padding_value=-100)
                    combined_student_batch[k] = batch_tensor
                else:
                    batch_tensor = pad_sequence(tensors, batch_first=True, padding_value=0)
                    combined_student_batch[k] = batch_tensor

            new_batch={"teacher_tokens":batch_teacher_tokens,
                       "student_tokens":batch_student_tokens,
                       'dpo_teacher_tokens': combined_teacher_batch,
                       'dpo_student_tokens': combined_student_batch,
                       "seq":torch.stack([torch.tensor(sample['seq']) for sample in batch], dim=0),
                       "cans":torch.stack([torch.tensor(sample['cans']) for sample in batch], dim=0),
                       "len_seq":torch.stack([torch.tensor(sample['len_seq']) for sample in batch], dim=0),
                       "len_cans":torch.stack([torch.tensor(sample['len_cans']) for sample in batch], dim=0),
                       "item_id": torch.stack([torch.tensor(sample['item_id']) for sample in batch], dim=0),
                       "flag":flag,
                       }
        else:
            batch_teacher_tokens = self.teacher_llm_tokenizer(
                inputs_text,
                return_tensors="pt",
                padding="longest",
                truncation=False,
                add_special_tokens=True,
                return_attention_mask=True)
            
            batch_student_tokens = self.student_llm_tokenizer(
                inputs_text,
                return_tensors="pt",
                padding="longest",
                truncation=False,
                add_special_tokens=True,
                return_attention_mask=True)
            
            cans_name=[sample['cans_name'] for sample in batch]
            
            new_batch={"teacher_tokens": batch_teacher_tokens,
                       "student_tokens": batch_student_tokens,
                       "seq":torch.stack([torch.tensor(sample['seq']) for sample in batch], dim=0),
                       "cans":torch.stack([torch.tensor(sample['cans']) for sample in batch], dim=0),
                       "len_seq":torch.stack([torch.tensor(sample['len_seq']) for sample in batch], dim=0),
                       "len_cans":torch.stack([torch.tensor(sample['len_cans']) for sample in batch], dim=0),
                       "item_id": torch.stack([torch.tensor(sample['item_id']) for sample in batch], dim=0),
                       "correct_answer": targets_text,
                       "cans_name": cans_name,
                       }
        return new_batch
    
class DInterface(pl.LightningDataModule):

    def __init__(self, 
                 llm_tokenizer=None,
                 num_workers=8,
                 dataset='',
                 **kwargs):
        super().__init__()
        self.num_workers = num_workers
        self.llm_tokenizer = llm_tokenizer
        self.dataset = dataset
        self.kwargs = kwargs
        self.batch_size = kwargs['batch_size']
        self.max_epochs = kwargs['max_epochs']
        self.load_data_module()
        self.load_prompt(kwargs['prompt_path'])

        self.trainset = self.instancialize(stage='train')
        self.valset = self.instancialize(stage='val')
        self.testset = self.instancialize(stage='test')
        self.max_steps = self.max_epochs * (len(self.trainset) // self.batch_size) // self.num_workers

    def train_dataloader(self):
        return DataLoader(self.trainset,
                          batch_size=self.batch_size, 
                          num_workers=self.num_workers, 
                          shuffle=True,
                          drop_last=True,
                          collate_fn=TrainCollater(prompt_list=self.prompt_list,llm_tokenizer=self.llm_tokenizer,train=True, max_step=self.max_steps))

    def val_dataloader(self):
        return DataLoader(self.valset, 
                          batch_size=self.batch_size, 
                          num_workers=self.num_workers, 
                          shuffle=False,
                          collate_fn=TrainCollater(prompt_list=self.prompt_list,llm_tokenizer=self.llm_tokenizer,train=False))

    def test_dataloader(self):
        return DataLoader(self.testset, 
                          batch_size=1, 
                          num_workers=self.num_workers, 
                          shuffle=False,
                          collate_fn=TrainCollater(prompt_list=self.prompt_list,llm_tokenizer=self.llm_tokenizer,train=False))

    def load_data_module(self):
        name = self.dataset
        camel_name = ''.join([i.capitalize() for i in name.split('_')])
        try:
            self.data_module = getattr(importlib.import_module(
                '.'+name, package=__package__), camel_name)
        except:
            raise ValueError(
                f'Invalid Dataset File Name or Invalid Class Name data.{name}.{camel_name}')

    def instancialize(self, **other_args):
        """ Instancialize a model using the corresponding parameters
            from self.hparams dictionary. You can also input any args
            to overwrite the corresponding value in self.kwargs.
        """
        class_args = inspect.getargspec(self.data_module.__init__).args[1:]
        inkeys = self.kwargs.keys()
        args1 = {}
        for arg in class_args:
            if arg in inkeys:
                args1[arg] = self.kwargs[arg]
        args1.update(other_args)
        return self.data_module(**args1)
    
    def load_prompt(self,prompt_path):
        if os.path.isfile(prompt_path):
            with open(prompt_path, 'r') as f:
                raw_prompts = f.read().splitlines()
            self.prompt_list = [p.strip() for p in raw_prompts]
            print('Load {} training prompts'.format(len(self.prompt_list)))
            print('Prompt Example \n{}'.format(random.choice(self.prompt_list)))
        else:
            self.prompt_list = []


class Distillation_DInterface(pl.LightningDataModule):

    def __init__(self, 
                 teacher_llm_tokenizer=None,
                 student_llm_tokenizer=None,
                 num_workers=8,
                 dataset='',
                 **kwargs):
        super().__init__()
        self.num_workers = num_workers
        self.teacher_llm_tokenizer = teacher_llm_tokenizer
        self.student_llm_tokenizer = student_llm_tokenizer
        self.dataset = dataset
        self.kwargs = kwargs
        self.batch_size = kwargs['batch_size']
        self.max_epochs = kwargs['max_epochs']
        self.load_data_module()
        self.load_prompt(kwargs['prompt_path'])

        self.trainset = self.instancialize(stage='train')
        self.valset = self.instancialize(stage='val')
        self.testset = self.instancialize(stage='test')
        self.max_steps = self.max_epochs * (len(self.trainset) // self.batch_size) // self.num_workers

    def train_dataloader(self):
        return DataLoader(self.trainset,
                          batch_size=self.batch_size, 
                          num_workers=self.num_workers, 
                          shuffle=True,
                          drop_last=True,
                          collate_fn=Distill_TrainCollater(prompt_list=self.prompt_list, teacher_llm_tokenizer=self.teacher_llm_tokenizer, student_llm_tokenizer=self.student_llm_tokenizer, train=True, max_step=self.max_steps))

    def val_dataloader(self):
        return DataLoader(self.valset, 
                          batch_size=1, 
                          num_workers=self.num_workers, 
                          shuffle=False,
                          collate_fn=Distill_TrainCollater(prompt_list=self.prompt_list, teacher_llm_tokenizer=self.teacher_llm_tokenizer, student_llm_tokenizer=self.student_llm_tokenizer, train=False))

    def test_dataloader(self):
        return DataLoader(self.testset, 
                          batch_size=1, 
                          num_workers=self.num_workers, 
                          shuffle=False,
                          collate_fn=Distill_TrainCollater(prompt_list=self.prompt_list, teacher_llm_tokenizer=self.teacher_llm_tokenizer, student_llm_tokenizer=self.student_llm_tokenizer, train=False))

    def load_data_module(self):
        name = self.dataset
        camel_name = ''.join([i.capitalize() for i in name.split('_')])
        try:
            self.data_module = getattr(importlib.import_module(
                '.'+name, package=__package__), camel_name)
        except:
            raise ValueError(
                f'Invalid Dataset File Name or Invalid Class Name data.{name}.{camel_name}')

    def instancialize(self, **other_args):
        """ Instancialize a model using the corresponding parameters
            from self.hparams dictionary. You can also input any args
            to overwrite the corresponding value in self.kwargs.
        """
        class_args = inspect.getargspec(self.data_module.__init__).args[1:]
        inkeys = self.kwargs.keys()
        args1 = {}
        for arg in class_args:
            if arg in inkeys:
                args1[arg] = self.kwargs[arg]
        args1.update(other_args)
        return self.data_module(**args1)
    
    def load_prompt(self,prompt_path):
        if os.path.isfile(prompt_path):
            with open(prompt_path, 'r') as f:
                raw_prompts = f.read().splitlines()
            self.prompt_list = [p.strip() for p in raw_prompts]
            print('Load {} training prompts'.format(len(self.prompt_list)))
            print('Prompt Example \n{}'.format(random.choice(self.prompt_list)))
        else:
            self.prompt_list = []