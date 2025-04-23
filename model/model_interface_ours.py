import inspect
import torch
import importlib
from torch import nn
from torch.nn import functional as F
import torch.optim.lr_scheduler as lrs
from torch.nn import CrossEntropyLoss
import pytorch_lightning as pl
from typing import List, Optional, Tuple, Union
from transformers import LlamaForCausalLM, LlamaTokenizer
import random
from pandas.core.frame import DataFrame
import os.path as op
import os
from optims import LinearWarmupCosineLRScheduler
import numpy as np
from peft import get_peft_config, get_peft_model, get_peft_model_state_dict, LoraConfig, TaskType, PeftModel, PeftConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from functools import wraps
from peft.utils import _get_batch_size
import types
from torch.optim.lr_scheduler import CosineAnnealingLR

class PeftModelForCausalLM(PeftModel):
    def __init__(
        self, model: torch.nn.Module, peft_config: PeftConfig, adapter_name: str = "default", **kwargs
    ) -> None:
        super().__init__(model, peft_config, adapter_name, **kwargs)
        self.base_model_prepare_inputs_for_generation = self.base_model.prepare_inputs_for_generation

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
        peft_config = self.active_peft_config
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


class PeftModelForCausalLM_prelim(PeftModel):
    def __init__(
        self, model: torch.nn.Module, peft_config: PeftConfig, adapter_name: str = "default", **kwargs
    ) -> None:
        super().__init__(model, peft_config, adapter_name, **kwargs)
        self.prelim_head = nn.Linear(4096, 32005, bias=False)
        self.prelim_norm = LlamaRMSNorm(4096) 
        self.base_model_prepare_inputs_for_generation = self.base_model.prepare_inputs_for_generation
        self.base_model.model.lm_head.register_forward_hook(self.hook1)
        # https://github.com/huggingface/transformers/blob/33868a057c02f0368ba63bd1edb746be38fe3d90/src/transformers/models/llama/modeling_llama.py#L865
        self.base_model.model.model.register_forward_hook(self.hook)

    def hook(self, module, input, output):
        all_hidden_states = output.hidden_states
        # 0, 1-32
        modified_last_hidden_state = self.prelim_norm(all_hidden_states[22])
        # modified_last_hidden_state = all_hidden_states[10]

        new_output = BaseModelOutputWithPast(
            last_hidden_state=modified_last_hidden_state,
            past_key_values=output.past_key_values,
            hidden_states=output.hidden_states,
            attentions=output.attentions
        )
        return new_output

    def hook1(self, module, input, output):
        if isinstance(input, tuple):
            new_output = self.prelim_head(input[0])
        else:
            new_output = self.prelim_head(input)
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
                    
class MInterface(pl.LightningModule):
    def __init__(self, 
                 **kargs):
        super().__init__()
        self.save_hyperparameters()
        self.load_llm(self.hparams.llm_path)
        self.load_rec_model(self.hparams.rec_model_path)
        self.load_projector()

    def forward(self, batch):
        targets = batch["tokens"].input_ids.masked_fill(
            batch["tokens"].input_ids == self.llama_tokenizer.pad_token_id, -100
        ) # [batch_size, max_len]
        targets = targets.masked_fill((batch["tokens"].token_type_ids == 0)[:,1:], -100)
        # targets = targets.masked_fill((batch["tokens"].token_type_ids == 0), -100)
        # input_embeds = self.wrap_emb(batch)
        input_embeds, user_sasrec_emb, _, pos_init_emb, _, pos_decode_emb, neg_init_emb, _, neg_decode_emb = self.wrap_emb(batch)

        outputs = self.llama_model(
            inputs_embeds=input_embeds,
            attention_mask=batch["tokens"].attention_mask,
            return_dict=True,
            labels=targets,
            use_cache=False,
            output_hidden_states=True
        )
        # return outputs
        pos_scores = (user_sasrec_emb * pos_decode_emb).sum(dim=1)
        neg_scores = (user_sasrec_emb * neg_decode_emb).sum(dim=1)

        pos_labels = torch.ones_like(pos_scores).to(pos_scores.device)  
        neg_labels = torch.zeros_like(neg_scores).to(neg_scores.device)  

        pos_loss = F.binary_cross_entropy_with_logits(pos_scores, pos_labels)
        neg_loss = F.binary_cross_entropy_with_logits(neg_scores, neg_labels)
        bce_loss = pos_loss + neg_loss

        MSE = nn.MSELoss(reduction='mean')
        mse_loss = MSE(pos_init_emb, pos_decode_emb) + MSE(neg_init_emb, neg_decode_emb)
        return outputs, mse_loss, bce_loss
        # return outputs

    def generate(self, batch,temperature=0.8,do_sample=False,num_beams=1,max_gen_length=64,min_gen_length=1,repetition_penalty=1.0,length_penalty=1.0, num_return_sequences=1):
        # input_embeds = self.wrap_emb(batch)
        input_embeds, _, _, _, _, _, _, _, _  = self.wrap_emb(batch)

        generate_ids = self.llama_model.generate(
            inputs_embeds=input_embeds,
            attention_mask=batch["tokens"].attention_mask,
            temperature=temperature,
            do_sample=do_sample,
            num_beams=num_beams,
            max_new_tokens=max_gen_length,
            min_new_tokens=min_gen_length,
            pad_token_id=self.llama_tokenizer.pad_token_id,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            num_return_sequences=num_return_sequences
        )
        output_text = self.llama_tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        outputs = [text.strip() for text in output_text]
        return outputs

    def training_step(self, batch, batch_idx):
        if self.scheduler:
            self.scheduler.step(self.trainer.global_step, self.current_epoch, self.trainer.max_steps)
        if batch["flag"]:
            for name, param in self.projector.named_parameters():
                param.requires_grad = False
            for name, param in self.projector1.named_parameters():
                param.requires_grad = False
            for name, param in self.decoder.named_parameters():
                param.requires_grad = False
        else:
            for name, param in self.projector.named_parameters():
                param.requires_grad = True
            for name, param in self.projector1.named_parameters():
                param.requires_grad = True
            for name, param in self.decoder.named_parameters():
                param.requires_grad = True
        # out = self(batch)

        out, mse_loss, bce_loss = self(batch)
        if random.random() < 0.5:
            loss = self.configure_loss(out) + mse_loss
        else:
            loss = self.configure_loss(out) + mse_loss + 0.5 * bce_loss
        # loss = self.configure_loss(out)
        self.log('bce_loss', bce_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('mse_loss', mse_loss, on_step=True, on_epoch=True, prog_bar=True)
        # loss = self.configure_loss(out)
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
        # optimizer = torch.optim.Adam([
        #     {'params': self.llama_model.parameters(), 'lr': self.hparams.lr}
        # ])
        optimizer = torch.optim.Adam([
            {'params': self.projector.parameters(), 'lr': self.hparams.lr, 'weight_decay':weight_decay},
            {'params': self.projector1.parameters(), 'lr': self.hparams.lr, 'weight_decay':weight_decay},
            {'params': self.decoder.parameters(), 'lr': self.hparams.lr, 'weight_decay':weight_decay},
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
                # self.scheduler = CosineAnnealingLR(optimizer, T_max=max_step * 15, eta_min=self.hparams.lr_decay_min_lr)
                self.scheduler = LinearWarmupCosineLRScheduler(optimizer,
                                                  max_step=max_step,
                                                  min_lr=self.hparams.lr_decay_min_lr,
                                                  init_lr=self.hparams.lr,
                                                  warmup_steps=warmup_steps,
                                                  warmup_start_lr=self.hparams.lr_warmup_start_lr)
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
                except AttributeError:
                    to_be_removed.append(key)
            for key in to_be_removed:
                checkpoint['state_dict'].pop(key)
        elif self.hparams.save == 'all':
            pass
        
    def load_llm(self, llm_path):
        print('Loading LLAMA')
        self.llama_tokenizer = LlamaTokenizer.from_pretrained(llm_path, use_fast=False)
        self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token
        self.llama_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.llama_tokenizer.padding_side = "right"
        # self.llama_tokenizer.add_special_tokens({'additional_special_tokens': ['[PH]','[HistoryEmb]','[CansEmb]','[ItemEmb]']})
        self.llama_tokenizer.add_special_tokens({'additional_special_tokens': ['[UserEmb]', '[PH]', '[HistoryEmb]','[CansEmb]','[ItemEmb]']})

        self.llama_model = LlamaForCausalLM.from_pretrained(llm_path, torch_dtype=torch.bfloat16)
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
                self.llama_model = PeftModelForCausalLM(self.llama_model, peft_config)
                # self.llama_model = PeftModelForCausalLM_prelim(self.llama_model, peft_config)
                # for name, param in self.llama_model.named_parameters():
                #     if 'filter' in name:
                #         param.requires_grad = True
                #     else:
                #         param.requires_grad = False    
            self.llama_model.print_trainable_parameters()
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
        self.projector1 = self.instancialize(Model, rec_size=self.hparams.rec_size, llm_size=self.llama_model.config.hidden_size)
        self.decoder = self.instancialize(Model, rec_size=self.llama_model.config.hidden_size, llm_size=self.hparams.rec_size)
       
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
        # return item_txt_embs
        item_decode_embs = self.decoder(item_txt_embs)
        return item_rec_embs, item_txt_embs, item_decode_embs
    
    def encode_user(self, seq, len_seq):
        if self.hparams.rec_embed=="SASRec":
            user_sasrec_embs = self.rec_model.cacul_h(seq, len_seq)
        user_rec_embs = self.projector1(user_sasrec_embs)
        return user_sasrec_embs, user_rec_embs
    
    def embed_tokens(self, token_ids):
        embeds = self.llama_model.base_model.embed_tokens(token_ids)
        return embeds

    def wrap_emb(self, batch):
        input_embeds = self.llama_model.get_input_embeddings()(batch["tokens"].input_ids)
        
        user_token_id = self.llama_tokenizer("[UserEmb]", return_tensors="pt",add_special_tokens=False).input_ids.item()
        his_token_id=self.llama_tokenizer("[HistoryEmb]", return_tensors="pt",add_special_tokens=False).input_ids.item()
        cans_token_id=self.llama_tokenizer("[CansEmb]", return_tensors="pt",add_special_tokens=False).input_ids.item()
        item_token_id=self.llama_tokenizer("[ItemEmb]", return_tensors="pt",add_special_tokens=False).input_ids.item()
        # his_item_embeds= self.encode_items(batch["seq"])
        # cans_item_embeds= self.encode_items(batch["cans"])
        # item_embeds=self.encode_items(batch["item_id"])
        
        user_sasrec_embeds, user_embeds = self.encode_user(batch["seq"], batch["len_seq"])
        _, his_item_embeds, _ = self.encode_items(batch["seq"])
        cans_init_embeds, cans_item_embeds, cands_decode_embeds = self.encode_items(batch["cans"])
        item_init_embeds, item_embeds, item_decode_embeds = self.encode_items(batch["item_id"])

        batch_size, _ = item_embeds.shape
        neg_init_embed = []
        neg_embed = []
        neg_decode_embed = []

        for i in range(batch_size):
            cans = batch["cans"][i]
            target_id = batch["item_id"][i]
            neg_indices = (cans != target_id).nonzero(as_tuple=True)[0]
            selected_index = random.choice(neg_indices.tolist())
            neg_init_embed.append(cans_init_embeds[i, selected_index])
            neg_embed.append(cans_item_embeds[i, selected_index])
            neg_decode_embed.append(cands_decode_embeds[i, selected_index])
        neg_embeds = torch.stack(neg_embed, dim=0)
        neg_init_embeds = torch.stack(neg_init_embed, dim=0)
        neg_decode_embeds = torch.stack(neg_decode_embed, dim=0)

        for i in range(len(batch["len_seq"])):
            if (batch["tokens"].input_ids[i] == user_token_id).nonzero().shape[0] > 0:
                idx_tensor=(batch["tokens"].input_ids[i] == user_token_id).nonzero().view(-1)
                for idx, item_emb in zip(idx_tensor, user_embeds[i,:batch["len_seq"][i].item()]):
                    input_embeds[i, idx] = item_emb
            if (batch["tokens"].input_ids[i]==his_token_id).nonzero().shape[0]>0:
                idx_tensor=(batch["tokens"].input_ids[i]==his_token_id).nonzero().view(-1)
                for idx, item_emb in zip(idx_tensor,his_item_embeds[i,:batch["len_seq"][i].item()]):
                    input_embeds[i,idx]=item_emb
            if (batch["tokens"].input_ids[i]==cans_token_id).nonzero().shape[0]>0:
                idx_tensor=(batch["tokens"].input_ids[i]==cans_token_id).nonzero().view(-1)
                for idx, item_emb in zip(idx_tensor,cans_item_embeds[i,:batch["len_cans"][i].item()]):
                    input_embeds[i,idx]=item_emb
            if (batch["tokens"].input_ids[i] == item_token_id).nonzero().shape[0]>0:
                idx=(batch["tokens"].input_ids[i] == item_token_id).nonzero().item()
                input_embeds[i,idx] = item_embeds[i]
        # return input_embeds
        return input_embeds, user_sasrec_embeds.squeeze(), user_embeds.squeeze(), item_init_embeds, item_embeds, item_decode_embeds, neg_init_embeds, neg_embeds, neg_decode_embeds
     
    def calculate_hr1(self, eval_content):
        correct_num=0
        valid_num=0
        total_num=0
        for i, generate in enumerate(eval_content["generate"]):
            real = eval_content["real"][i]
            cans = eval_content["cans"][i]
            total_num += 1
            generate = generate.strip().lower().strip()
            real=real.strip().lower().strip()
            cans=[item.strip().lower().strip() for item in cans]
            gen_cans_list=[]
            for cans_item in cans:
                if cans_item in generate:
                    gen_cans_list.append(cans_item)
            if len(gen_cans_list)==1:
                valid_num+=1
                if real == gen_cans_list[0]:
                    correct_num+=1
        valid_ratio=valid_num/total_num
        if valid_num>0:
            hr1=correct_num/valid_num
        else:
            hr1=0
        return valid_ratio,hr1