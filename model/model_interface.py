import inspect
import torch
import sys
import importlib
import pickle
from torch import nn
from torch.nn import functional as F
import torch.optim.lr_scheduler as lrs
from torch.nn import CrossEntropyLoss
import pytorch_lightning as pl
from typing import List, Optional, Tuple, Union
from transformers import LlamaForCausalLM, LlamaTokenizer
from pandas.core.frame import DataFrame
import os.path as op
import os
from optims import LinearWarmupCosineLRScheduler
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.llama.modeling_llama import LlamaRMSNorm
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, BitsAndBytesConfig
from typing import Any, Dict, List, Optional, Union
from peft import TaskType, PeftModel, PeftConfig, LoraConfig
from sklearn.decomposition import PCA

def pad_to_length(tensor: torch.Tensor, length: int, pad_value: Union[int, float], dim: int = -1) -> torch.Tensor:
    if tensor.size(dim) >= length:
        return tensor
    else:
        pad_size = list(tensor.shape)
        pad_size[dim] = length - tensor.size(dim)
        return torch.cat(
            [
                tensor,
                pad_value * torch.ones(*pad_size, dtype=tensor.dtype, device=tensor.device),
            ],
            dim=dim,
        )

class PeftModelForCausalLM(PeftModel):
    def __init__(
        self, model: torch.nn.Module, peft_config: PeftConfig, adapter_name: str = "default", **kwargs
    ) -> None:
        super().__init__(model, peft_config, adapter_name, **kwargs)
        self.base_model_prepare_inputs_for_generation = self.base_model.prepare_inputs_for_generation
        # https://github.com/huggingface/transformers/blob/33868a057c02f0368ba63bd1edb746be38fe3d90/src/transformers/models/llama/modeling_llama.py#L865

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        task_ids=None,
        **kwargs,
    ):
        with self._enable_peft_forward_hooks(**kwargs):
            kwargs = {k: v for k, v in kwargs.items() if k not in self.special_peft_forward_args}
            outputs = self.base_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    inputs_embeds=inputs_embeds,
                    labels=labels,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                    **kwargs,
                )
            return outputs

    def generate(self, *args, **kwargs):
        self.base_model.prepare_inputs_for_generation = self.prepare_inputs_for_generation
        if hasattr(self.base_model, "model"):
            self.base_model.model.generation_config = self.generation_config
        else:
            self.base_model.generation_config = self.generation_config
       
        with self._enable_peft_forward_hooks(*args, **kwargs):
            kwargs = {k: v for k, v in kwargs.items() if k not in self.special_peft_forward_args}
            outputs = self.base_model.generate(*args, **kwargs)
       
        return outputs

    def prepare_inputs_for_generation(self, *args, task_ids: Optional[torch.Tensor] = None, **kwargs):
        model_kwargs = self.base_model_prepare_inputs_for_generation(*args, **kwargs)
        return model_kwargs
        
# https://github.com/huggingface/transformers/blob/33868a057c02f0368ba63bd1edb746be38fe3d90/src/transformers/models/llama/modeling_llama.py#L865
class PeftModelForCausalLM_prelim(PeftModel):
    def __init__(
        self, model: torch.nn.Module, peft_config: PeftConfig, stage: str = None, **kwargs
    ) -> None:
        super().__init__(model, peft_config, **kwargs)
        #32005
        self.stage = stage
        out_dim = 128261
        # out_dim = 32005
        # out_dim = 50270

        self.prelim_norm = LlamaRMSNorm(2048) 
        self.base_model_prepare_inputs_for_generation = self.base_model.prepare_inputs_for_generation
        if self.stage == 'distillation_stage_prelim':
            self.hidden_states_cache = None
            self.base_model.model.model.register_forward_hook(self.store_hidden_states)
            self.base_model.model.lm_head.register_forward_hook(self.hook0)
        elif self.stage == 'distillation_stage_1':
            self.prelim_head = nn.Linear(2048, out_dim, bias=False)
            self.prelim_filter1 = nn.Linear(4096, 2048, bias=False)
            self.prelim_filter2 = nn.Linear(4096, 2048, bias=False)
            self.prelim_filter3 = nn.Linear(4096, 2048, bias=False)
            self.prelim_filter4 = nn.Linear(4096, 2048, bias=False)
            self.base_model.model.model.register_forward_hook(self.hook)
        elif self.stage == 'distillation_stage2':
            self.prelim_filter1 = nn.Linear(4096, 2048, bias=False)
            self.prelim_filter2 = nn.Linear(4096, 2048, bias=False)
            self.prelim_filter3 = nn.Linear(4096, 2048, bias=False)
            self.prelim_filter4 = nn.Linear(4096, 2048, bias=False)
            self.base_model.model.model.register_forward_hook(self.hook1)
    
    def store_hidden_states(self, module, input, output):
        self.hidden_states_cache = output.hidden_states 

    def hook0(self, module, input, output):
        hidden_state = self.hidden_states_cache[11]
        logits = self.prelim_head(hidden_state)
        return logits
    
    def hook(self, module, input, output):
        all_hidden_states = list(output.hidden_states)

        all_hidden_states[24] = self.prelim_head(self.prelim_norm(self.prelim_filter1(all_hidden_states[24])))
        all_hidden_states[26] = self.prelim_head(self.prelim_norm(self.prelim_filter2(all_hidden_states[26])))
        all_hidden_states[28] = self.prelim_head(self.prelim_norm(self.prelim_filter3(all_hidden_states[28])))
        all_hidden_states[30] = self.prelim_head(self.prelim_norm(self.prelim_filter4(all_hidden_states[30])))

        new_output = BaseModelOutputWithPast(
            last_hidden_state=output.last_hidden_state,
            past_key_values=output.past_key_values,
            hidden_states=tuple(all_hidden_states),
            attentions=output.attentions
        )
        return new_output
    
    def hook1(self, module, input, output):
        all_hidden_states = list(output.hidden_states)
        
        all_hidden_states[24] = self.prelim_filter1(all_hidden_states[24])
        all_hidden_states[26] = self.prelim_filter2(all_hidden_states[26])
        all_hidden_states[28] = self.prelim_filter3(all_hidden_states[28])
        all_hidden_states[30] = self.prelim_filter4(all_hidden_states[30])

        new_output = BaseModelOutputWithPast(
            last_hidden_state=output.last_hidden_state,
            past_key_values=output.past_key_values,
            hidden_states=tuple(all_hidden_states),
            attentions=output.attentions
        )
        return new_output

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        task_ids=None,
        **kwargs,
    ):
        with self._enable_peft_forward_hooks(**kwargs):
            kwargs = {k: v for k, v in kwargs.items() if k not in self.special_peft_forward_args}
            outputs = self.base_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    inputs_embeds=inputs_embeds,
                    labels=labels,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                    **kwargs,)
            return outputs
        
    def generate(self, *args, **kwargs):
        self.base_model.prepare_inputs_for_generation = self.prepare_inputs_for_generation
        if hasattr(self.base_model, "model"):
            self.base_model.model.generation_config = self.generation_config
        else:
            self.base_model.generation_config = self.generation_config
        
        with self._enable_peft_forward_hooks(*args, **kwargs):
            kwargs = {k: v for k, v in kwargs.items() if k not in self.special_peft_forward_args}
            outputs = self.base_model.generate(*args, **kwargs)  
        return outputs

    def prepare_inputs_for_generation(self, *args, task_ids: Optional[torch.Tensor] = None, **kwargs):
        model_kwargs = self.base_model_prepare_inputs_for_generation(*args, **kwargs)
        return model_kwargs
    
class MInterface(pl.LightningModule):
    def __init__(self, 
                 **kargs):
        super().__init__()
        self.save_hyperparameters()
        self.load_llm(self.hparams.llm_path)
        self.load_rec_model(self.hparams.rec_model_path)
        self.load_projector()    
    
    def forward(self, batch):
        targets = batch["teacher_tokens"].input_ids.masked_fill(
            batch["teacher_tokens"].input_ids == self.llama_tokenizer.pad_token_id, -100
        ) # [batch_size, max_len]
        batch["teacher_tokens"].token_type_ids = torch.nn.functional.pad(batch["teacher_tokens"].token_type_ids, (0, 2), value=1)
        targets = targets.masked_fill((batch["teacher_tokens"].token_type_ids == 0)[:,:], -100)

        input_embeds = self.wrap_emb(batch)
        outputs = self.llama_model(
            inputs_embeds=input_embeds,
            attention_mask=batch["teacher_tokens"].attention_mask,
            return_dict=True,
            labels=targets,
            output_hidden_states=True,
            use_cache=False
        )
        return outputs
    
    
    def generate(self, batch, temperature=0.8,do_sample=False,num_beams=1,max_gen_length=64,min_gen_length=1,repetition_penalty=1.0,length_penalty=1.0, num_return_sequences=1):
        input_embeds = self.wrap_emb(batch)
        generate_ids = self.llama_model.generate(
            inputs_embeds=input_embeds,
            attention_mask=batch["teacher_tokens"].attention_mask,
            temperature=temperature,
            do_sample=do_sample,
            num_beams=num_beams,
            max_new_tokens=max_gen_length,
            min_new_tokens=min_gen_length,
            pad_token_id=self.llama_tokenizer.pad_token_id,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            num_return_sequences=num_return_sequences,
            output_hidden_states=True,
            )
        output_text=self.llama_tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        outputs=[text.strip() for text in output_text]
        return outputs

    def training_step(self, batch, batch_idx):
        if self.scheduler:
            self.scheduler.step()
            # self.scheduler.step(self.trainer.global_step, self.current_epoch, self.trainer.max_steps)
        if batch["flag"]:
            for name, param in self.projector.named_parameters():
                param.requires_grad = False
        else:
            for name, param in self.projector.named_parameters():
                param.requires_grad = True
        out = self(batch)
        loss = self.configure_loss(out)
        self.log('loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('lr', self.scheduler.optimizer.param_groups[0]['lr'], on_step=True, on_epoch=True, prog_bar=True)
        self.log('global_step_num', self.trainer.global_step, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def on_validation_epoch_start(self):
        self.val_content={
            "generate":[],
            "real":[],
            "cans":[],
        }

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        generate_output = self.generate(batch)
        output=[]
        for i,generate in enumerate(generate_output):
            real=batch['correct_answer'][i]
            cans=batch['cans_name'][i]
            generate=generate.strip().split("\n")[0]
            output.append((generate,real,cans))
        return output

    def on_validation_batch_end(self, outputs, batch, batch_idx, dataloader_idx):
        for generate,real,cans in outputs:
            self.val_content["generate"].append(generate)
            self.val_content["real"].append(real)
            self.val_content["cans"].append(cans)

    def on_validation_epoch_end(self):
        df=DataFrame(self.val_content)
        if not os.path.exists(self.hparams.output_dir):
            os.makedirs(self.hparams.output_dir)
        df.to_csv(op.join(self.hparams.output_dir, 'valid.csv'))
        prediction_valid_ratio,hr=self.calculate_hr1(self.val_content)
        metric=hr*prediction_valid_ratio
        self.log('val_prediction_valid', prediction_valid_ratio, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_hr', hr, on_step=False, on_epoch=True, prog_bar=True)
        self.log('metric', metric, on_step=False, on_epoch=True, prog_bar=True)

    def on_test_epoch_start(self):
        self.test_content={
            "generate":[],
            "real":[],
            "cans":[],
        }

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        generate_output = self.generate(batch)
        output=[]
        for i,generate in enumerate(generate_output):
            real=batch['correct_answer'][i]
            cans=batch['cans_name'][i]
            generate=generate.strip().split("\n")[0]
            if generate == real:
                print('generate real', generate, real)
            output.append((generate,real,cans))
        return output
    
    def on_test_batch_end(self, outputs, batch, batch_idx, dataloader_idx):
        for generate,real,cans in outputs:
            self.test_content["generate"].append(generate)
            self.test_content["real"].append(real)
            self.test_content["cans"].append(cans)

    def on_test_epoch_end(self):
        df=DataFrame(self.test_content)
        if not os.path.exists(self.hparams.output_dir):
            os.makedirs(self.hparams.output_dir)
        df.to_csv(op.join(self.hparams.output_dir, 'test.csv'))
        prediction_valid_ratio,hr=self.calculate_hr1(self.test_content)
        metric=hr*prediction_valid_ratio
        self.log('test_prediction_valid', prediction_valid_ratio, on_step=False, on_epoch=True, prog_bar=True)
        self.log('test_hr', hr, on_step=False, on_epoch=True, prog_bar=True)
        self.log('metric', metric, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        if hasattr(self.hparams, 'weight_decay'):
            weight_decay = self.hparams.weight_decay
        else:
            weight_decay = 0
        optimizer = torch.optim.Adam([
            {'params': self.projector.parameters(), 'lr': self.hparams.lr, 'weight_decay':weight_decay},
            {'params': self.llama_model.parameters(), 'lr': self.hparams.lr}
        ])

        if self.hparams.lr_scheduler is None:
            return optimizer
        else:
            max_step = self.trainer.max_steps
            warmup_steps = max_step // 20
            print(f'max_step: {max_step}')
            print(f'warmup_steps: {warmup_steps}')
            if self.hparams.lr_scheduler == 'cosine':
                self.scheduler = CosineAnnealingLR(optimizer, T_max=max_step * 15, eta_min=self.hparams.lr_decay_min_lr)
                # self.scheduler = LinearWarmupCosineLRScheduler(optimizer,
                #                                   max_step=max_step,
                #                                   min_lr=self.hparams.lr_decay_min_lr,
                #                                   init_lr=self.hparams.lr,
                #                                   warmup_steps=warmup_steps,
                #                                   warmup_start_lr=self.hparams.lr_warmup_start_lr)
            else:
                self.scheduler = None
                raise ValueError('Invalid lr_scheduler type!')
            return optimizer

    def configure_loss(self, out, labels=None):
        loss = self.hparams.loss.lower()
        if loss == 'lm':
            return out.loss
        else:
            raise ValueError("Invalid Loss Type!")
        
    def on_save_checkpoint(self, checkpoint):
        if self.hparams.save == 'part':
            checkpoint.pop('optimizer_states')
            to_be_removed = []
            for key, value in checkpoint['state_dict'].items():
                try:
                    if not self.get_parameter(key).requires_grad:
                        to_be_removed.append(key)
                    if 'prelim_head' in key:
                        to_be_removed.append(key)
                except AttributeError:
                    to_be_removed.append(key)
            for key in to_be_removed:
                checkpoint['state_dict'].pop(key)
            
            projector_state_dict = self.projector.state_dict()
            checkpoint['state_dict'].update(projector_state_dict)
        elif self.hparams.save == 'all':
            pass
        
    def load_llm(self, llm_path):
        print('Loading LLAMA')
        # self.llama_tokenizer = LlamaTokenizer.from_pretrained(llm_path, use_fast=False)
        self.llama_tokenizer = AutoTokenizer.from_pretrained(llm_path, use_fast=False, trust_remote_code=True,)
        self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token
        self.llama_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.llama_tokenizer.padding_side = "right"
        self.llama_tokenizer.add_special_tokens({'additional_special_tokens': ['[PH]','[HistoryEmb]','[CansEmb]','[ItemEmb]']})

        # quantization_config = BitsAndBytesConfig(
        #     load_in_4bit=True, 
        #     bnb_4bit_compute_dtype=torch.float16, 
        # )

        # self.llama_model = AutoModelForCausalLM.from_pretrained(llm_path, low_cpu_mem_usage=True)        
        self.llama_model = AutoModelForCausalLM.from_pretrained(llm_path, device_map="auto")

        # self.llama_model = LlamaForCausalLM.from_pretrained(llm_path, torch_dtype=torch.bfloat16)
        self.llama_model.resize_token_embeddings(len(self.llama_tokenizer))
        
        if self.hparams.llm_tuning == 'lora':
            if self.hparams.peft_dir:
                self.llama_model = PeftModel.from_pretrained(self.llm_model, self.hparams.peft_dir, is_trainable=True)
            else:
                if self.hparams.peft_config:
                    peft_config = LoraConfig(**LoraConfig.from_json_file(self.hparams.peft_config))
                else:
                    peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM,
                                             inference_mode=False,
                                             r=self.hparams.lora_r,
                                             lora_alpha=self.hparams.lora_alpha,
                                             lora_dropout=self.hparams.lora_dropout,
                                             target_modules=['k_proj', 'v_proj', 'q_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'])
                self.peft_config = peft_config
                self.llama_model = PeftModelForCausalLM(self.llama_model, peft_config)                

            self.llama_model.print_trainable_parameters()
        elif self.hparams.llm_tuning == 'freeze':
            for name, param in self.llama_model.named_parameters():
                param.requires_grad = False
       
        else:
            raise NotImplementedError()
 
        print('Loading LLAMA Done')

    def load_projector(self):
        name = self.hparams.model_name
        camel_name = ''.join([i.capitalize() for i in name.split('_')])
        try:
            Model = getattr(importlib.import_module(
                '.'+name, package=__package__), camel_name)
        except:
            raise ValueError(
                f'Invalid Module File Name or Invalid Class Name {name}.{camel_name}!')
        self.projector = self.instancialize(Model, rec_size=self.hparams.rec_size, llm_size=self.llama_model.config.hidden_size)

    def instancialize(self, Model, **other_args):
        class_args = inspect.getargspec(Model.__init__).args[1:]
        inkeys = self.hparams.keys()
        args1 = {}
        for arg in class_args:
            if arg in inkeys:
                args1[arg] = getattr(self.hparams, arg)
        args1.update(other_args)
        return Model(**args1)

    def load_rec_model(self, rec_model_path):
        print('Loading Rec Model')
        self.rec_model = torch.load(rec_model_path, map_location="cpu")
        self.rec_model.eval()
        for name, param in self.rec_model.named_parameters():
            param.requires_grad = False
        print('Loding Rec model Done')

    def encode_items(self, seq):
        if self.hparams.rec_embed=="SASRec":
            item_rec_embs=self.rec_model.cacu_x(seq)
        elif self.hparams.rec_embed in ['Caser','GRU']:
            item_rec_embs=self.rec_model.item_embeddings(seq)
        item_txt_embs=self.projector(item_rec_embs)
        return item_txt_embs
    
    def embed_tokens(self, token_ids):
        embeds = self.llama_model.base_model.embed_tokens(token_ids)
        return embeds

    def wrap_emb(self, batch):
        input_embeds = self.llama_model.get_input_embeddings()(batch["teacher_tokens"].input_ids)
        
        his_token_id=self.llama_tokenizer("[HistoryEmb]", return_tensors="pt",add_special_tokens=False).input_ids.item()
        cans_token_id=self.llama_tokenizer("[CansEmb]", return_tensors="pt",add_special_tokens=False).input_ids.item()
        item_token_id=self.llama_tokenizer("[ItemEmb]", return_tensors="pt",add_special_tokens=False).input_ids.item()
        his_item_embeds= self.encode_items(batch["seq"])
        cans_item_embeds= self.encode_items(batch["cans"])
        item_embeds=self.encode_items(batch["item_id"])
            
        for i in range(len(batch["len_seq"])):
            if (batch["teacher_tokens"].input_ids[i]==his_token_id).nonzero().shape[0]>0:
                idx_tensor=(batch["teacher_tokens"].input_ids[i]==his_token_id).nonzero().view(-1)
                for idx, item_emb in zip(idx_tensor,his_item_embeds[i,:batch["len_seq"][i].item()]):
                    input_embeds[i,idx]=item_emb
            if (batch["teacher_tokens"].input_ids[i]==cans_token_id).nonzero().shape[0]>0:
                idx_tensor=(batch["teacher_tokens"].input_ids[i]==cans_token_id).nonzero().view(-1)
                for idx, item_emb in zip(idx_tensor,cans_item_embeds[i,:batch["len_cans"][i].item()]):
                    input_embeds[i,idx]=item_emb
            if (batch["teacher_tokens"].input_ids[i]==item_token_id).nonzero().shape[0]>0:
                idx=(batch["teacher_tokens"].input_ids[i]==item_token_id).nonzero().item()
                input_embeds[i,idx]=item_embeds[i]
        return input_embeds
    
    def get_mv_title(self, s):
        sub_list=[", The", ", A", ", An"]
        for sub_s in sub_list:
            if sub_s in s:
                return sub_s[2:]+" "+s.replace(sub_s,"")
        return s
     
    def calculate_hr1(self,eval_content):
        correct_num=0
        valid_num=0
        total_num=0
        
        # pop_correct = 0
        # lesspop_correct = 0
        # true_pop = 0
        # true_lesspop = 0

        # with open('/data/mcy/LLaRA/hf_data/lastfm/popular_group.pkl', 'rb') as pklfile:
        #     popular_group_loaded = pickle.load(pklfile)
        # with open('/data/mcy/LLaRA/hf_data/lastfm/less_popular_group.pkl', 'rb') as pklfile:
        #     less_popular_group_loaded = pickle.load(pklfile)
        
        # pop_keys = [key.lower() for key in popular_group_loaded.keys()]
        # less_pop_keys = [key.lower() for key in less_popular_group_loaded.keys()]

        for i,generate in enumerate(eval_content["generate"]):
            real = eval_content["real"][i]
            cans = eval_content["cans"][i]
            total_num += 1
            generate = generate.strip().lower().strip()
            real = real.strip().lower().strip()
            cans = [item.strip().lower().strip() for item in cans]
            gen_cans_list = []
            
            # if real in pop_keys:
            #     true_pop += 1
            # elif real in less_pop_keys:
            #     true_lesspop += 1

            for cans_item in cans:
                if cans_item in generate:
                    gen_cans_list.append(cans_item)

            if len(gen_cans_list) == 1:
                valid_num += 1
                if real == gen_cans_list[0]:
                    correct_num += 1
                    # if real.lower() in pop_keys:
                    #     pop_correct += 1
                    # elif real.lower() in less_pop_keys:
                    #     lesspop_correct += 1
        
        # print('correct num, pop, lesspop', correct_num, true_pop, true_lesspop, pop_correct, lesspop_correct)
        valid_ratio = valid_num / total_num
        if valid_num > 0:
            hr1 = correct_num / valid_num
        else:
            hr1 = 0
        return valid_ratio,hr1

class MLPProjector(nn.Module):
    def __init__(self, rec_size, hidden_size=2048):
        super().__init__()
        self.layer1 = nn.Linear(rec_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.ReLU()
        # self.layer1 = nn.Linear(rec_size, hidden_size)
            
    def forward(self, x):
        x = self.activation(self.layer1(x))
        x = self.layer2(x)
        # x = self.layer1(x)
        return x
            

class KDLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.IGNORE_INDEX = -100

    def forward(self, logits: torch.Tensor, teacher_logits: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        temperature = 1.0
        teacher_probs = F.softmax(teacher_logits / temperature, dim=-1, dtype=torch.float32)
        inf_mask = torch.isinf(logits)
        logprobs = F.log_softmax(logits / temperature, dim=-1, dtype=torch.float32)
        prod_probs = torch.masked_fill(teacher_probs * logprobs, inf_mask, 0)
        
        x = torch.sum(prod_probs, dim=-1).view(-1)
        mask = (label != self.IGNORE_INDEX).int()
        loss = -torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)

        return loss

class Router(nn.Module):
    def __init__(self, num_experts, k):
        super().__init__()
        self.num_experts = num_experts
        self.k = k
    
    def forward(self, student_features, teacher_features: list):
        student_features_norm = F.normalize(student_features, dim=-1)        
        cosine_sims = []
        for i in range(self.num_experts):
            teacher_features_norm = F.normalize(teacher_features[i], dim=-1)
            sim = torch.sum(student_features_norm * teacher_features_norm, dim=(1, 2))  
            cosine_sims.append(sim)
        
        cosine_sim = torch.stack(cosine_sims, dim=-1)
        _, max_sim_indices = torch.max(cosine_sim, dim=-1)  # [batch_size]
        teacher_features_with_new_dim = [tensor.unsqueeze(0) for tensor in teacher_features]
        concatenated_teacher = torch.cat(teacher_features_with_new_dim, dim=0)
        
        batch_size, seq_len, feature_dim = student_features.size()
        max_sim_indices = max_sim_indices.view(1, batch_size, 1, 1)  
        selected_teacher_features = torch.gather(concatenated_teacher, 0, max_sim_indices.expand(1, batch_size, seq_len, feature_dim))
        selected_teacher_features = selected_teacher_features.squeeze(0)
        return selected_teacher_features

class GlobalZScoreNormalizer:
    def __init__(self, momentum=0.99, eps=1e-6):
        self.momentum = momentum  # 滑动均值的动量
        self.eps = eps  # 避免数值不稳定
        self.global_mean = None  # 全局均值
        self.global_std = None  # 全局标准差
    
    def update_and_normalize(self, teacher_gap):
        batch_mean = teacher_gap.mean()
        batch_std = teacher_gap.std()

        # 初始化全局均值和标准差（仅第一次）
        if self.global_mean is None:
            self.global_mean = batch_mean
            self.global_std = batch_std
        else:
            # 采用指数移动平均更新全局统计量
            self.global_mean = self.momentum * self.global_mean + (1 - self.momentum) * batch_mean
            self.global_std = self.momentum * self.global_std + (1 - self.momentum) * batch_std

        # 归一化并计算 difficulty_weight
        normalized_teacher_gap = (teacher_gap - self.global_mean) / (self.global_std + self.eps)
        weight = torch.sigmoid(normalized_teacher_gap)  

        return weight

class DistillationInterface(pl.LightningModule):
    def __init__(self, 
                 **kargs):
        super().__init__()
        self.save_hyperparameters()
        if self.hparams.stage == 'distillation_stage_prelim':
            self.load_teacher_llm(self.hparams.llm_path)
            self.load_projector()
        elif self.hparams.stage == 'distillation_stage1':
            self.load_teacher_llm(self.hparams.llm_path)
            self.load_projector()
        elif self.hparams.stage == 'distillation_stage2':
            self.load_teacher_llm(self.hparams.llm_path)
            self.load_student_llm(self.hparams.student_pretrained_path)
            self.load_projector()
            self.load_student_projector()
            self.feat_mlp = nn.Linear(2048, 4096)
            self.feat_mlp1 = nn.Linear(2048, 2048)

        elif self.hparams.stage == 'distillation_stage3':
            self.load_teacher_llm(self.hparams.llm_path)
            self.load_student_llm(self.hparams.student_pretrained_path)
            self.load_projector()
            self.load_student_projector()
            self.global_normalizer = GlobalZScoreNormalizer(momentum=0.99)

        self.load_rec_model(self.hparams.rec_model_path)
        self.label_pad_token_id = -100

    @staticmethod
    def concatenated_inputs(
        batch: Dict[str, Union[List, torch.LongTensor]],
        label_pad_token_id: int = -100,
        padding_value: int = 0,
        device: Optional[torch.device] = None,
    ) -> Dict[str, torch.LongTensor]:
  
        concatenated_batch = {}
        max_length = max(batch["chosen_input_ids"].shape[1], batch["rejected_input_ids"].shape[1])

        for k in batch:
            if k.startswith("chosen") and isinstance(batch[k], torch.Tensor):
                if "labels" in k:
                    pad_value = label_pad_token_id
                elif k.endswith("_input_ids"):
                    pad_value = padding_value
                elif k.endswith("_attention_mask"):
                    pad_value = 0
                concatenated_key = k.replace("chosen", "concatenated")
                concatenated_batch[concatenated_key] = pad_to_length(batch[k], max_length, pad_value=pad_value)
        for k in batch:
            if k.startswith("rejected") and isinstance(batch[k], torch.Tensor):
                if "labels" in k:
                    pad_value = label_pad_token_id
                elif k.endswith("_input_ids"):
                    pad_value = padding_value
                elif k.endswith("_attention_mask"):
                    pad_value = 0
                concatenated_key = k.replace("rejected", "concatenated")
                concatenated_batch[concatenated_key] = torch.cat(
                    (
                        concatenated_batch[concatenated_key],
                        pad_to_length(batch[k], max_length, pad_value=pad_value),
                    ),
                    dim=0,
                ).to(device=device)
        return concatenated_batch
        
    def _get_batch_logps(
        self,
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
        average_log_prob: bool = False,
    ) -> torch.FloatTensor:
        """Compute the log probabilities of the given labels under the given logits.

        Args:
            logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
            labels: Labels for which to compute the log probabilities. Label tokens with a value of label_pad_token_id are ignored. Shape: (batch_size, sequence_length)
            average_log_prob: If True, return the average log probability per (non-masked) token. Otherwise, return the sum of the log probabilities of the (non-masked) tokens.

        Returns:
            A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the given logits.
        """
        if logits.shape[:-1] != labels.shape:
            raise ValueError("Logits (batch and sequence length dim) and labels must have the same shape.")

        labels = labels[:, 1:].clone()
        logits = logits[:, :-1, :]
        loss_mask = labels != self.label_pad_token_id

        labels[labels == self.label_pad_token_id] = 0

        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

        if average_log_prob:
            return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
        else:
            return (per_token_logps * loss_mask).sum(-1)
        
    def concat_teacher_forward(self, batch):
        concatenated_batch = self.concatenated_inputs(batch['dpo_teacher_tokens'])
        concat_embed = self.wrap_concat_teacher_emb(concatenated_batch, batch)
        all_logits = self.teacher_llama_model(
            inputs_embeds=concat_embed,
            attention_mask=concatenated_batch["concatenated_attention_mask"],
            use_cache=False,
            output_hidden_states=True,
        ).logits.to(torch.float32)

        len_chosen = batch['dpo_teacher_tokens']["chosen_labels"].shape[0]

        all_logps = self._get_batch_logps(
            all_logits,
            concatenated_batch["concatenated_labels"],
            average_log_prob=False,
        )
      
        chosen_logps = all_logps[:len_chosen]
        rejected_logps = all_logps[len_chosen:]

        chosen_logits = all_logits[:len_chosen]
        rejected_logits = all_logits[len_chosen:]

        chosen_labels = concatenated_batch["concatenated_labels"][:len_chosen]

        return (chosen_logps, rejected_logps, chosen_logits, rejected_logits, chosen_labels)
    
    def concat_student_forward(self, batch):
        concatenated_batch = self.concatenated_inputs(batch['dpo_student_tokens'])
        concat_embed = self.wrap_concat_student_emb(concatenated_batch, batch)
        all_logits = self.student_llama_model(
            inputs_embeds=concat_embed,
            attention_mask=concatenated_batch["concatenated_attention_mask"],
            output_hidden_states=True,
            use_cache=False
        ).logits.to(torch.float32)

        len_chosen = batch['dpo_student_tokens']["chosen_labels"].shape[0]

        all_logps = self._get_batch_logps(
            all_logits,
            concatenated_batch["concatenated_labels"],
            average_log_prob=False,
        )

        chosen_logps = all_logps[:len_chosen]
        rejected_logps = all_logps[len_chosen:]

        chosen_logits = all_logits[:len_chosen]
        rejected_logits = all_logits[len_chosen:]

        chosen_labels = concatenated_batch["concatenated_labels"][:len_chosen]

        return (chosen_logps, rejected_logps, chosen_logits, rejected_logits, chosen_labels)

    def teacher_forward(self, batch):
        targets = batch["teacher_tokens"].input_ids.masked_fill(
            batch["teacher_tokens"].input_ids == self.teacher_llama_tokenizer.pad_token_id, -100) 
        targets = targets.masked_fill((batch["teacher_tokens"].token_type_ids == 0)[:,:], -100)
        input_embeds = self.wrap_teacher_emb(batch)
        outputs = self.teacher_llama_model(
            inputs_embeds=input_embeds,
            attention_mask=batch["teacher_tokens"].attention_mask,
            return_dict=True,
            labels=targets,
            output_hidden_states=True,
            use_cache=False)
        if self.hparams.stage == 'distillation_stage_prelim':
            return outputs 

        elif self.hparams.stage == 'distillation_stage1':
            criterion = CrossEntropyLoss()
            labels_flat = targets.view(-1)  

            hidden_24 = outputs.hidden_states[24]
            hidden_26 = outputs.hidden_states[26]
            hidden_28 = outputs.hidden_states[28]
            hidden_30 = outputs.hidden_states[30]
            
            hidden_24_flat = hidden_24.view(-1, hidden_24.size(-1)) 
            hidden_26_flat = hidden_26.view(-1, hidden_26.size(-1)) 
            hidden_28_flat = hidden_30.view(-1, hidden_28.size(-1)) 
            hidden_30_flat = hidden_30.view(-1, hidden_30.size(-1)) 
            
            loss = criterion(hidden_24_flat, labels_flat) + criterion(hidden_26_flat, labels_flat) + criterion(hidden_28_flat, labels_flat)+criterion(hidden_30_flat, labels_flat)
            outputs.loss = loss
        return outputs
 

    def student_forward(self, batch):
        targets = batch["student_tokens"].input_ids.masked_fill(
            batch["student_tokens"].input_ids == self.student_llama_tokenizer.pad_token_id, -100) # [batch_size, max_len]
        targets = targets.masked_fill((batch["student_tokens"].token_type_ids == 0)[:,:], -100)
        input_embeds = self.wrap_student_emb(batch)

        outputs = self.student_llama_model(
            inputs_embeds=input_embeds,
            attention_mask=batch["student_tokens"].attention_mask,
            return_dict=True,
            labels=targets,
            output_hidden_states=True,
            use_cache=False
        )
        return outputs 
    
    def teacher_generate(self, batch,temperature=0.8,do_sample=False,num_beams=1,max_gen_length=64,min_gen_length=1,repetition_penalty=1.0,length_penalty=1.0, num_return_sequences=1):
        input_embeds = self.wrap_teacher_emb(batch)
        generate_ids = self.teacher_llama_model.generate(
            inputs_embeds=input_embeds,
            attention_mask=batch["teacher_tokens"].attention_mask,
            temperature=temperature,
            do_sample=do_sample,
            num_beams=num_beams,
            max_new_tokens=max_gen_length,
            min_new_tokens=min_gen_length,
            pad_token_id=self.teacher_llama_tokenizer.pad_token_id,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            num_return_sequences=num_return_sequences,
            output_hidden_states=True,
        )
        output_text=self.teacher_llama_tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        outputs=[text.strip() for text in output_text]
        return outputs
    
    def student_generate(self, batch, temperature=0.8, do_sample=False, num_beams=1, max_gen_length=64, min_gen_length=1, repetition_penalty=1.0, length_penalty=1.0, num_return_sequences=1):
        input_embeds = self.wrap_student_emb(batch)
        generate_ids = self.student_llama_model.generate(
            inputs_embeds=input_embeds,
            attention_mask=batch["student_tokens"].attention_mask,
            temperature=temperature,
            do_sample=do_sample,
            num_beams=num_beams,
            max_new_tokens=max_gen_length,
            min_new_tokens=min_gen_length,
            pad_token_id=self.student_llama_tokenizer.pad_token_id,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            num_return_sequences=num_return_sequences,
            )
        output_text=self.student_llama_tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        outputs=[text.strip() for text in output_text]
        return outputs
    
    def calc_hidden_loss(self, teacher_state, student_state, alpha=0.5):
        student_state = F.normalize(student_state, p=2, dim=-1)
        teacher_state = F.normalize(teacher_state, p=2, dim=-1)
        l2_loss = F.mse_loss(student_state, teacher_state)
        cos_loss = 1 - F.cosine_similarity(student_state.view(-1), teacher_state.view(-1), dim=0)    
        total_loss = alpha * l2_loss + (1 - alpha) * cos_loss
        return total_loss
    
    # def dpo_loss(self, policy_chosen_logps: torch.FloatTensor, policy_rejected_logps: torch.FloatTensor, reference_chosen_logps: torch.FloatTensor, reference_rejected_logps: torch.FloatTensor, reference_free: bool = False,) -> torch.FloatTensor:
        
    #     beta = 0.5
      
    #     chosen_logratios = policy_chosen_logps - reference_chosen_logps
    #     print(f"chosen:{chosen_logratios}")
    #     rejected_logratios = policy_rejected_logps - reference_rejected_logps
    #     print(f"reject:{rejected_logratios}")

    #     temp1 = beta * (chosen_logratios - rejected_logratios)
    #     losses = -F.logsigmoid(temp1)
    #     return losses.mean()
    
    def dpo_loss(self, policy_chosen_logps: torch.FloatTensor,
                policy_rejected_logps: torch.FloatTensor, reference_chosen_logps: torch.FloatTensor,
                reference_rejected_logps: torch.FloatTensor, reference_free: bool = False,) -> torch.FloatTensor:
        beta_base = 0.5
    
        chosen_logratios = policy_chosen_logps - reference_chosen_logps
        rejected_logratios = policy_rejected_logps - reference_rejected_logps
    
        # # ================= 困难样本识别 =================
        # # 情况1：学生与老师chosen差距小的样本（值越小越困难）
        student_teacher_diff = chosen_logratios  # 该值越小说明学生表现越差
        difficulty_weight1 = 1 - torch.sigmoid(student_teacher_diff)  # 困难样本权重
    
        # 情况2：老师自身chosen-rejected差距小的样本（值越小说明样本越模糊）
        teacher_gap = (reference_chosen_logps - reference_rejected_logps)
        difficulty_weight2 = self.global_normalizer.update_and_normalize(teacher_gap)

        beta_dynamic = beta_base * difficulty_weight2
        
        log_odds = chosen_logratios - rejected_logratios
        losses = -F.logsigmoid(beta_dynamic * log_odds) * difficulty_weight1
        print(difficulty_weight1, difficulty_weight2)
        total_loss = losses.mean()
        return total_loss
    
    def reduce_dimensions(self, weights, n_components=2):
        pca = PCA(n_components=n_components)
        reduced = pca.fit_transform(weights)
        return reduced
    
    def training_step(self, batch, batch_idx):
        if self.scheduler:
            # self.scheduler.step(self.trainer.global_step, self.current_epoch, self.trainer.max_steps)
            self.scheduler.step()
        if batch["flag"]:
            for name, param in self.projector.named_parameters():
                param.requires_grad = False
        else:
            for name, param in self.projector.named_parameters():
                param.requires_grad = False
        
        if self.hparams.stage == 'distillation_stage_prelim':     
            out_teacher = self.teacher_forward(batch)
            loss = self.configure_loss(out_teacher)

        elif self.hparams.stage == 'distillation_stage1':
            out_teacher = self.teacher_forward(batch)
            loss = self.configure_loss(out_teacher)

        elif self.hparams.stage == 'distillation_stage2':
            for name, param in self.projector.named_parameters():
                param.requires_grad = False

            for name, param in self.student_projector.named_parameters():
                param.requires_grad = True
            
            for name, param in self.feat_mlp.named_parameters():
                param.requires_grad = True
            
            for name, param in self.feat_mlp1.named_parameters():
                param.requires_grad = True
                
            kd_loss = KDLoss()
            out_teacher = self.teacher_forward(batch)
            out_student = self.student_forward(batch)
            # llama3 33 17/23
            teacher_hidden_states = [state for i, state in enumerate(out_teacher.hidden_states) if i in [30]]
            student_hidden_states = [state for i, state in enumerate(out_student.hidden_states) if i in [15]]

            distill_loss = 0
            # gated_teacher_state = self.router(student_hidden_states[0], teacher_hidden_states)
            for teacher_state, student_state in zip(teacher_hidden_states, student_hidden_states):
                distill_loss += self.calc_hidden_loss(teacher_state, student_state, alpha=0.5)

            labels = batch["student_tokens"].input_ids.masked_fill(batch["student_tokens"].input_ids == self.student_llama_tokenizer.pad_token_id, -100)
            labels = labels.masked_fill((batch["student_tokens"].token_type_ids == 0)[:,:], -100)
            student_logit = self.teacher_llama_model.base_model.model.lm_head(self.feat_mlp(student_hidden_states[0]))
            logit_loss = kd_loss(student_logit, out_teacher.logits, labels)

            # teacher_logit = self.student_llama_model.base_model.model.lm_head(self.feat_mlp1(teacher_hidden_states[0]))
            # logit_loss1 = kd_loss(teacher_logit, out_student.logits, labels)

            weight = self.feat_mlp.weight
            weight = weight / (torch.norm(weight, dim=1, keepdim=True) + 1e-6)
            weight_trans = weight.permute(1, 0)
            ones = torch.eye(weight.size(0)).to(weight.device)
            ones2 = torch.eye(weight.size(1)).to(weight.device)
            orth_loss = torch.dist(torch.mm(weight, weight_trans), ones, p=2)/ weight.numel() + torch.dist(torch.mm(weight_trans, weight), ones2, p=2)/ weight.numel()

            loss = self.configure_loss(out_student) + 0.5 * distill_loss +  0.5 * (logit_loss) + 0.05 * orth_loss  
            

        # elif self.hparams.stage == 'distillation_stage3':
        #     for name, param in self.projector.named_parameters():
        #         param.requires_grad = False

        #     for name, param in self.student_projector.named_parameters():
        #         param.requires_grad = True

        #     (policy_chosen_logps, policy_rejected_logps, _, _, _,) = self.concat_student_forward(batch)
        #     with torch.no_grad():
        #         (
        #         reference_chosen_logps,
        #         reference_rejected_logps,
        #         _,
        #         _,_,) = self.concat_teacher_forward(batch)

        #     # dpo_loss = self.dpo_loss(policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps,)
        #     beta_base = 0.5
        #     chosen_logratios = policy_chosen_logps - reference_chosen_logps
        #     rejected_logratios = policy_rejected_logps - reference_rejected_logps
        #     # # ================= 困难样本识别 =================
        #     # # 情况1：学生与老师chosen差距小的样本（值越小越困难）
        #     # student_teacher_diff = chosen_logratios  # 该值越小说明学生表现越差
        #     # difficulty_weight_1 = 1 - torch.sigmoid(student_teacher_diff)  # 困难样本权重
        #     # 情况2：老师自身chosen-rejected差距小的样本（值越小说明样本越模糊）
        #     teacher_gap = (reference_chosen_logps - reference_rejected_logps)
        #     difficulty_weight = torch.sigmoid(teacher_gap)  # 老师模糊样本权重
        #     difficulty_weight = self.global_normalizer.update_and_normalize(teacher_gap)
        #     log_odds = chosen_logratios - rejected_logratios - 0.5 * difficulty_weight
        #     losses = -F.logsigmoid(beta_base * (log_odds)) 
        #     dpo_loss = losses.mean()
        #     out_student = self.student_forward(batch)
        #     loss = self.configure_loss(out_student) + dpo_loss
        
        if self.hparams.stage == 'distillation_stage2':
            self.log('logit_loss', logit_loss, on_step=True, on_epoch=True, prog_bar=True)
            self.log('distill_loss', distill_loss, on_step=True, on_epoch=True, prog_bar=True)
            self.log('loss', loss, on_step=True, on_epoch=True, prog_bar=True)
            self.log('orth loss', orth_loss, on_step=True, on_epoch=True, prog_bar=True)
        # if self.hparams.stage == 'distillation_stage3':
        #     self.log('dpo_loss', dpo_loss, on_step=True, on_epoch=True, prog_bar=True)
        
        self.log('lr', self.scheduler.optimizer.param_groups[1]['lr'], on_step=True, on_epoch=True, prog_bar=True)
        self.log('global_step_num', self.trainer.global_step, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    

    def on_validation_epoch_start(self):
        self.val_content={
            "generate":[],
            "real":[],
            "cans":[],
        }

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        if self.hparams.stage == 'distillation_stage1' or self.hparams.stage == 'distillation_stage_prelim':
            generate_output = self.teacher_generate(batch)
        else:
            generate_output = self.student_generate(batch)
        output=[]
        for i,generate in enumerate(generate_output):
            real=batch['correct_answer'][i]
            cans=batch['cans_name'][i]
            generate=generate.strip().split("\n")[0]
            output.append((generate,real,cans))
        return output

    def on_validation_batch_end(self, outputs, batch, batch_idx, dataloader_idx):
        for generate,real,cans in outputs:
            self.val_content["generate"].append(generate)
            self.val_content["real"].append(real)
            self.val_content["cans"].append(cans)

    def on_validation_epoch_end(self):
        df=DataFrame(self.val_content)
        if not os.path.exists(self.hparams.output_dir):
            os.makedirs(self.hparams.output_dir)
        df.to_csv(op.join(self.hparams.output_dir, 'valid.csv'))
        prediction_valid_ratio, hr = self.calculate_hr1(self.val_content)
        metric = hr*prediction_valid_ratio
        self.log('val_prediction_valid', prediction_valid_ratio, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_hr', hr, on_step=False, on_epoch=True, prog_bar=True)
        self.log('metric', metric, on_step=False, on_epoch=True, prog_bar=True)

    def on_test_epoch_start(self):
        self.test_content={
            "generate":[],
            "real":[],
            "cans":[],
        }

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        if self.hparams.stage == 'distillation_stage_prelim' or self.hparams.stage == 'distillation_stage1':
            generate_output = self.teacher_generate(batch)
        else:
            generate_output = self.student_generate(batch)
        for param in self.feat_mlp.parameters():
            param.requires_grad = False
        for param in self.student_projector.parameters():
            param.requires_grad = False
        output=[]
        for i, generate in enumerate(generate_output):
            real = batch['correct_answer'][i]
            cans = batch['cans_name'][i]
            generate = generate.strip().split("\n")[0]
            # generate = " ".join(generate.strip().split("\n")[:3])
            if generate == real:
                print('predict', generate, real)
            output.append((generate,real,cans))
        return output
    
    def on_test_batch_end(self, outputs, batch, batch_idx, dataloader_idx):
        for generate,real,cans in outputs:
            self.test_content["generate"].append(generate)
            self.test_content["real"].append(real)
            self.test_content["cans"].append(cans)

    def on_test_epoch_end(self):
        df=DataFrame(self.test_content)
        if not os.path.exists(self.hparams.output_dir):
            os.makedirs(self.hparams.output_dir)
        df.to_csv(op.join(self.hparams.output_dir, 'test.csv'))
        prediction_valid_ratio,hr=self.calculate_hr1(self.test_content)
        metric=hr*prediction_valid_ratio
        self.log('test_prediction_valid', prediction_valid_ratio, on_step=False, on_epoch=True, prog_bar=True)
        self.log('test_hr', hr, on_step=False, on_epoch=True, prog_bar=True)
        self.log('metric', metric, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        if hasattr(self.hparams, 'weight_decay'):
            weight_decay = self.hparams.weight_decay
        else:
            weight_decay = 0

        if self.hparams.stage == 'distillation_stage1' or self.hparams.stage == 'distillation_stage_prelim':
            optimizer = torch.optim.Adam([
                {'params': self.teacher_llama_model.parameters(), 'lr': self.hparams.lr}
            ])
        elif self.hparams.stage == 'distillation_stage2':
            optimizer = torch.optim.AdamW([
            {'params': self.feat_mlp.parameters(), 'lr': self.hparams.lr, 'weight_decay':weight_decay},
            {'params': self.student_projector.parameters(), 'lr': self.hparams.lr, 'weight_decay':weight_decay},
            {'params': self.student_llama_model.parameters(), 'lr': self.hparams.lr}
            ])
            # upper_optimizer = torch.optim.Adam([
            # {'params': self.feat_mlp.parameters(), 'lr': self.hparams.lr * 0.1, 'weight_decay':weight_decay},
            # ])
            # optimizer = [lower_optimizer, upper_optimizer]
        elif self.hparams.stage == 'distillation_stage3':
            optimizer = torch.optim.Adam([
                {'params': self.student_llama_model.parameters(), 'lr': self.hparams.lr},
                {'params': self.student_projector.parameters(), 'lr': self.hparams.lr, 'weight_decay':weight_decay}
            ])
        
      
        if self.hparams.lr_scheduler is None:
            return optimizer
        else:
            max_step = self.trainer.max_steps
            warmup_steps = max_step // 20
            print(f'max_step: {max_step}')
            print(f'warmup_steps: {warmup_steps}')
            if self.hparams.lr_scheduler == 'cosine':
                self.scheduler = CosineAnnealingLR(optimizer, T_max=max_step * 10, eta_min=self.hparams.lr_decay_min_lr)
              
            else:
                self.scheduler = None
                raise ValueError('Invalid lr_scheduler type!')
            return optimizer

    def configure_loss(self, out, labels=None):
        loss = self.hparams.loss.lower()
        if loss == 'lm':
            return out.loss
        else:
            raise ValueError("Invalid Loss Type!")
        
    def on_save_checkpoint(self, checkpoint):
        if self.hparams.save == 'part':
            checkpoint.pop('optimizer_states')
            to_be_removed = []
            for key, value in checkpoint['state_dict'].items():
                try:
                    if not self.get_parameter(key).requires_grad:
                        to_be_removed.append(key)
                    elif 'router' in key and self.hparams.stage == 'distillation_stage2':
                        to_be_removed.append(key)
                except AttributeError:
                    to_be_removed.append(key)
            for key in to_be_removed:
                checkpoint['state_dict'].pop(key)
            
        elif self.hparams.save == 'all':
            pass
        
    def load_teacher_llm(self, llm_path):
        self.teacher_llama_tokenizer = AutoTokenizer.from_pretrained(llm_path, use_fast=False)
        # self.teacher_llama_tokenizer = LlamaTokenizer.from_pretrained(llm_path, use_fast=False)
        self.teacher_llama_tokenizer.pad_token = self.teacher_llama_tokenizer.eos_token
        self.teacher_llama_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.teacher_llama_tokenizer.padding_side = "right"
        self.teacher_llama_tokenizer.add_special_tokens({'additional_special_tokens': ['[PH]','[HistoryEmb]','[CansEmb]','[ItemEmb]']})
        # self.teacher_llama_model = LlamaForCausalLM.from_pretrained(llm_path, torch_dtype=torch.bfloat16)
        print('Loading teacher model')
        self.teacher_llama_model = AutoModelForCausalLM.from_pretrained(llm_path)
        self.teacher_llama_model.resize_token_embeddings(len(self.teacher_llama_tokenizer))
      
        if self.hparams.llm_tuning == 'lora':
            peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM,
                                             inference_mode=False,
                                             r=self.hparams.lora_r,
                                             lora_alpha=self.hparams.lora_alpha,
                                             lora_dropout=self.hparams.lora_dropout,
                                             target_modules=['k_proj', 'v_proj', 'q_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'])
            self.peft_config = peft_config                
            # self.teacher_llama_model = PeftModelForCausalLM(self.teacher_llama_model, peft_config)
            
            if self.hparams.stage == 'distillation_stage_prelim':
                self.teacher_llama_model = PeftModelForCausalLM_prelim(self.teacher_llama_model, peft_config, 'distillation_stage_prelim')
                for name, param in self.teacher_llama_model.named_parameters():
                    if 'prelim' in name:
                        param.requires_grad = True
                    else:
                        param.requires_grad = False
            elif self.hparams.stage == 'distillation_stage1':
                self.teacher_llama_model = PeftModelForCausalLM_prelim(self.teacher_llama_model, peft_config, 'distillation_stage1')
                for name, param in self.teacher_llama_model.named_parameters():
                    if 'prelim' in name:
                        param.requires_grad = True
                    else:
                        param.requires_grad = False
            elif self.hparams.stage == 'distillation_stage2':
                self.teacher_llama_model = PeftModelForCausalLM_prelim(self.teacher_llama_model, peft_config, 'distillation_stage2')
                self.teacher_llama_model.eval() 
                for name, param in self.teacher_llama_model.named_parameters():
                    param.requires_grad = False  

            elif self.hparams.stage == 'distillation_stage3':
                self.teacher_llama_model = PeftModelForCausalLM_prelim(self.teacher_llama_model, peft_config)
                self.teacher_llama_model.eval()  
                for name, param in self.teacher_llama_model.named_parameters():
                    param.requires_grad = False 
            self.teacher_llama_model.print_trainable_parameters()
 
        print('Loading teacherLLAMA Done')

    def load_student_llm(self, model_path):
        print('Loading student LLAMA')
        # self.student_llama_tokenizer = LlamaTokenizer.from_pretrained(model_path, use_fast=False)
        self.student_llama_tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        self.student_llama_tokenizer.pad_token = self.student_llama_tokenizer.eos_token
        self.student_llama_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.student_llama_tokenizer.padding_side = "right"
        self.student_llama_tokenizer.add_special_tokens({'additional_special_tokens': ['[PH]','[HistoryEmb]','[CansEmb]','[ItemEmb]']})
        # self.student_llama_model = LlamaForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)
        self.student_llama_model = AutoModelForCausalLM.from_pretrained(model_path)
        self.student_llama_model.resize_token_embeddings(len(self.student_llama_tokenizer))
      
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=['k_proj', 'v_proj', 'q_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
            inference_mode=False,
            r=self.hparams.lora_r,
            lora_alpha=self.hparams.lora_alpha,
            lora_dropout=self.hparams.lora_dropout,
        )
        self.peft_config = peft_config

        self.student_llama_model = PeftModelForCausalLM(self.student_llama_model, peft_config)
        for name, param in self.student_llama_model.named_parameters():
            if 'lora' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        self.student_llama_model.print_trainable_parameters()
        print('Loading student LLAMA Done')

    def load_projector(self):
        name = self.hparams.model_name
        camel_name = ''.join([i.capitalize() for i in name.split('_')])
        try:
            Model = getattr(importlib.import_module(
                '.'+name, package=__package__), camel_name)
        except:
            raise ValueError(
                f'Invalid Module File Name or Invalid Class Name {name}.{camel_name}!')
        self.projector = self.instancialize(Model, rec_size=self.hparams.rec_size, llm_size=self.teacher_llama_model.config.hidden_size)
        for param in self.projector.parameters():
            param.requires_grad = False
            
    def load_student_projector(self):
        name = self.hparams.model_name
        camel_name = ''.join([i.capitalize() for i in name.split('_')])
        try:
            Model = getattr(importlib.import_module(
                '.'+name, package=__package__), camel_name)
        except:
            raise ValueError(
                f'Invalid Module File Name or Invalid Class Name {name}.{camel_name}!')
        self.student_projector = self.instancialize(Model, rec_size=self.hparams.rec_size, llm_size=self.student_llama_model.config.hidden_size)
        for param in self.student_projector.parameters():
            param.requires_grad = True
            
    def instancialize(self, Model, **other_args):
        class_args = inspect.getargspec(Model.__init__).args[1:]
        inkeys = self.hparams.keys()
        args1 = {}
        for arg in class_args:
            if arg in inkeys:
                args1[arg] = getattr(self.hparams, arg)
        args1.update(other_args)
        return Model(**args1)

    def load_rec_model(self, rec_model_path):
        print('Loading Rec Model')
        self.rec_model = torch.load(rec_model_path, map_location="cpu")
        self.rec_model.eval()
        for name, param in self.rec_model.named_parameters():
            param.requires_grad = False
        print('Loding Rec model Done')

    def encode_items(self, seq):
        if self.hparams.rec_embed=="SASRec":
            item_rec_embs=self.rec_model.cacu_x(seq)
        elif self.hparams.rec_embed in ['Caser','GRU']:
            item_rec_embs=self.rec_model.item_embeddings(seq)
        item_txt_embs=self.projector(item_rec_embs)
        return item_txt_embs
    
    def encode_users(self, seq, len_seq):
        if self.hparams.rec_embed=="SASRec":
            user_rec_embs=self.rec_model.cacul_h(seq, len_seq)
        elif self.hparams.rec_embed in ['Caser','GRU']:
            user_rec_embs=self.rec_model.item_embeddings(seq)
        
        user_txt_embs=self.projector(user_rec_embs)    
        return user_txt_embs
    
    def encode_items_student(self, seq):
        if self.hparams.rec_embed=="SASRec":
            item_rec_embs=self.rec_model.cacu_x(seq)
        elif self.hparams.rec_embed in ['Caser','GRU']:
            item_rec_embs=self.rec_model.item_embeddings(seq)
        item_txt_embs=self.student_projector(item_rec_embs)
        return item_txt_embs
    
    def embed_tokens(self, token_ids):
        embeds = self.teacher_llama_model.base_model.embed_tokens(token_ids)
        return embeds

    def wrap_teacher_emb(self, batch):
        input_embeds = self.teacher_llama_model.get_input_embeddings()(batch["teacher_tokens"].input_ids)
        
        his_token_id=self.teacher_llama_tokenizer("[HistoryEmb]", return_tensors="pt",add_special_tokens=False).input_ids.item()
        cans_token_id=self.teacher_llama_tokenizer("[CansEmb]", return_tensors="pt",add_special_tokens=False).input_ids.item()
        item_token_id=self.teacher_llama_tokenizer("[ItemEmb]", return_tensors="pt",add_special_tokens=False).input_ids.item()
        his_item_embeds= self.encode_items(batch["seq"])
        cans_item_embeds= self.encode_items(batch["cans"])
        item_embeds=self.encode_items(batch["item_id"])
            
        for i in range(len(batch["len_seq"])):
            if (batch["teacher_tokens"].input_ids[i]==his_token_id).nonzero().shape[0]>0:
                idx_tensor=(batch["teacher_tokens"].input_ids[i]==his_token_id).nonzero().view(-1)
                for idx, item_emb in zip(idx_tensor,his_item_embeds[i,:batch["len_seq"][i].item()]):
                    input_embeds[i,idx]=item_emb
            if (batch["teacher_tokens"].input_ids[i]==cans_token_id).nonzero().shape[0]>0:
                idx_tensor=(batch["teacher_tokens"].input_ids[i]==cans_token_id).nonzero().view(-1)
                for idx, item_emb in zip(idx_tensor,cans_item_embeds[i,:batch["len_cans"][i].item()]):
                    input_embeds[i,idx]=item_emb
            if (batch["teacher_tokens"].input_ids[i]==item_token_id).nonzero().shape[0]>0:
                idx=(batch["teacher_tokens"].input_ids[i]==item_token_id).nonzero().item()
                input_embeds[i,idx]=item_embeds[i]
        return input_embeds

    def wrap_student_emb(self, batch):
        input_embeds = self.student_llama_model.get_input_embeddings()(batch["student_tokens"].input_ids)
        
        his_token_id=self.student_llama_tokenizer("[HistoryEmb]", return_tensors="pt",add_special_tokens=False).input_ids.item()
        cans_token_id=self.student_llama_tokenizer("[CansEmb]", return_tensors="pt",add_special_tokens=False).input_ids.item()
        item_token_id=self.student_llama_tokenizer("[ItemEmb]", return_tensors="pt",add_special_tokens=False).input_ids.item()
        his_item_embeds= self.encode_items_student(batch["seq"])
        cans_item_embeds= self.encode_items_student(batch["cans"])
        item_embeds=self.encode_items_student(batch["item_id"])
        # user_embeds=self.encode_users_student(batch["seq"], batch["len_seq"])
    
        for i in range(len(batch["len_seq"])):
            if (batch["student_tokens"].input_ids[i]==his_token_id).nonzero().shape[0]>0:
                idx_tensor=(batch["student_tokens"].input_ids[i]==his_token_id).nonzero().view(-1)
                for idx, item_emb in zip(idx_tensor,his_item_embeds[i,:batch["len_seq"][i].item()]):
                    input_embeds[i,idx]=item_emb
            if (batch["student_tokens"].input_ids[i]==cans_token_id).nonzero().shape[0]>0:
                idx_tensor=(batch["student_tokens"].input_ids[i]==cans_token_id).nonzero().view(-1)
                for idx, item_emb in zip(idx_tensor,cans_item_embeds[i,:batch["len_cans"][i].item()]):
                    input_embeds[i,idx]=item_emb
            if (batch["student_tokens"].input_ids[i]==item_token_id).nonzero().shape[0]>0:
                idx=(batch["student_tokens"].input_ids[i]==item_token_id).nonzero().item()
                input_embeds[i,idx]=item_embeds[i]
        return input_embeds
    
    def wrap_concat_teacher_emb(self, concat_batch, batch):
        input_embeds = self.teacher_llama_model.get_input_embeddings()(concat_batch["concatenated_input_ids"])
        
        his_token_id = self.teacher_llama_tokenizer("[HistoryEmb]", return_tensors="pt",add_special_tokens=False).input_ids.item()
        cans_token_id = self.teacher_llama_tokenizer("[CansEmb]", return_tensors="pt",add_special_tokens=False).input_ids.item()
        item_token_id = self.teacher_llama_tokenizer("[ItemEmb]", return_tensors="pt",add_special_tokens=False).input_ids.item()
        his_item_embeds = self.encode_items(torch.cat((batch["seq"], batch["seq"]), dim=0))
        cans_item_embeds = self.encode_items(torch.cat((batch["cans"], batch["cans"]), dim=0))
        item_embeds = self.encode_items(torch.cat((batch["item_id"], batch["item_id"]), dim=0))

        concat_len_seq = torch.cat((batch["len_seq"], batch["len_seq"]), dim=0)
        concat_len_cans = torch.cat((batch["len_cans"], batch["len_cans"]), dim=0)    
    
        for i in range(len(concat_len_seq)):
            if (concat_batch["concatenated_input_ids"][i]==his_token_id).nonzero().shape[0]>0:
                idx_tensor=(concat_batch["concatenated_input_ids"][i]==his_token_id).nonzero().view(-1)
                for idx, item_emb in zip(idx_tensor,his_item_embeds[i,:concat_len_seq[i].item()]):
                    input_embeds[i,idx]=item_emb
            if (concat_batch["concatenated_input_ids"][i]==cans_token_id).nonzero().shape[0]>0:
                idx_tensor=(concat_batch["concatenated_input_ids"][i]==cans_token_id).nonzero().view(-1)
                for idx, item_emb in zip(idx_tensor,cans_item_embeds[i,:concat_len_cans[i].item()]):
                    input_embeds[i,idx]=item_emb
            if (concat_batch["concatenated_input_ids"][i]==item_token_id).nonzero().shape[0]>0:
                idx=(concat_batch["concatenated_input_ids"][i]==item_token_id).nonzero().item()
                input_embeds[i,idx]=item_embeds[i]
        return input_embeds
    
    def wrap_concat_student_emb(self, concat_batch, batch):
        input_embeds = self.student_llama_model.get_input_embeddings()(concat_batch["concatenated_input_ids"])
        
        his_token_id = self.student_llama_tokenizer("[HistoryEmb]", return_tensors="pt",add_special_tokens=False).input_ids.item()
        cans_token_id = self.student_llama_tokenizer("[CansEmb]", return_tensors="pt",add_special_tokens=False).input_ids.item()
        item_token_id = self.student_llama_tokenizer("[ItemEmb]", return_tensors="pt",add_special_tokens=False).input_ids.item()
        his_item_embeds = self.encode_items_student(torch.cat((batch["seq"], batch["seq"]), dim=0))
        cans_item_embeds = self.encode_items_student(torch.cat((batch["cans"], batch["cans"]), dim=0))
        item_embeds = self.encode_items_student(torch.cat((batch["item_id"], batch["item_id"]), dim=0))

        concat_len_seq = torch.cat((batch["len_seq"], batch["len_seq"]), dim=0)
        concat_len_cans = torch.cat((batch["len_cans"], batch["len_cans"]), dim=0)    
    
        for i in range(len(concat_len_seq)):
            if (concat_batch["concatenated_input_ids"][i]==his_token_id).nonzero().shape[0]>0:
                idx_tensor=(concat_batch["concatenated_input_ids"][i]==his_token_id).nonzero().view(-1)
                for idx, item_emb in zip(idx_tensor,his_item_embeds[i,:concat_len_seq[i].item()]):
                    input_embeds[i,idx]=item_emb
            if (concat_batch["concatenated_input_ids"][i]==cans_token_id).nonzero().shape[0]>0:
                idx_tensor=(concat_batch["concatenated_input_ids"][i]==cans_token_id).nonzero().view(-1)
                for idx, item_emb in zip(idx_tensor,cans_item_embeds[i,:concat_len_cans[i].item()]):
                    input_embeds[i,idx]=item_emb
            if (concat_batch["concatenated_input_ids"][i]==item_token_id).nonzero().shape[0]>0:
                idx=(concat_batch["concatenated_input_ids"][i]==item_token_id).nonzero().item()
                input_embeds[i,idx]=item_embeds[i]
        return input_embeds
    
    def get_mv_title(self, s):
        sub_list=[", The", ", A", ", An"]
        for sub_s in sub_list:
            if sub_s in s:
                return sub_s[2:]+" "+s.replace(sub_s,"")
        return s
     
    def calculate_hr1(self,eval_content):
        correct_num=0
        valid_num=0
        total_num=0
        import unicodedata

        def normalize_string(s):
            return unicodedata.normalize("NFKC", s).strip().lower()

        for i,generate in enumerate(eval_content["generate"]):
            real = eval_content["real"][i]
            cans = eval_content["cans"][i]
            total_num += 1
            generate = normalize_string(generate.strip().lower().strip())
            real = real.strip().lower().strip()
            cans = [normalize_string(item.strip().lower().strip()) for item in cans]
            gen_cans_list = []

            for cans_item in cans:
                if cans_item in generate or cans_item in [generate]:
                    gen_cans_list.append(cans_item)

            if len(gen_cans_list) == 1:
                valid_num += 1
                if real == gen_cans_list[0]:
                    correct_num += 1
                   
        valid_ratio = valid_num / total_num
        if valid_num > 0:
            hr1 = correct_num / valid_num
        else:
            hr1 = 0
        return valid_ratio,hr1
    