import warnings
from collections import defaultdict
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union
import importlib


import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset
from transformers import DataCollator, PreTrainedModel, PreTrainedTokenizerBase, Trainer, TrainingArguments
from transformers.trainer_callback import TrainerCallback

from .utils import DPODataCollatorWithPadding, pad_to_length


def is_peft_available():
    return importlib.util.find_spec("peft") is not None

if is_peft_available():
    from peft import get_peft_model, prepare_model_for_kbit_training
    # from peft import get_peft_model, prepare_model_for_int8_training


class DPODistillTrainer(Trainer):
    r"""
    Initialize DPOTrainer.

    Args:
        model (`transformers.PreTrainedModel`):
            The model to train, preferably an `AutoModelForSequenceClassification`.
        ref_model (`PreTrainedModelWrapper`):
            Hugging Face transformer model with a casual language modelling head. Used for implicit reward computation and loss.
        beta (`float`, defaults to 0.1):
            The beta factor in DPO loss. Higher beta means less divergence from the initial policy.
        args (`transformers.TrainingArguments`):
            The arguments to use for training.
        data_collator (`transformers.DataCollator`):
            The data collator to use for training. If None is specified, the default data collator (`DPODataCollatorWithPadding`) will be used
            which will pad the sequences to the maximum length of the sequences in the batch, given a dataset of paired sequences.
        label_pad_token_id (`int`, defaults to `-100`):
            The label pad token id. This argument is required if you want to use the default data collator.
        padding_value (`int`, defaults to `0`):
            The padding value. This argument is required if you want to use the default data collator.
        truncation_mode (`str`, defaults to `keep_end`):
            The truncation mode to use, either `keep_end` or `keep_start`. This argument is required if you want to use the default data collator.
        train_dataset (`datasets.Dataset`):
            The dataset to use for training.
        eval_dataset (`datasets.Dataset`):
            The dataset to use for evaluation.
        tokenizer (`transformers.PreTrainedTokenizerBase`):
            The tokenizer to use for training. This argument is required if you want to use the default data collator.
        model_init (`Callable[[], transformers.PreTrainedModel]`):
            The model initializer to use for training. If None is specified, the default model initializer will be used.
        callbacks (`List[transformers.TrainerCallback]`):
            The callbacks to use for training.
        optimizers (`Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`):
            The optimizer and scheduler to use for training.
        preprocess_logits_for_metrics (`Callable[[torch.Tensor, torch.Tensor], torch.Tensor]`):
            The function to use to preprocess the logits before computing the metrics.
        max_length (`int`, defaults to `None`):
            The maximum length of the sequences in the batch. This argument is required if you want to use the default data collator.
        max_prompt_length (`int`, defaults to `None`):
            The maximum length of the prompt. This argument is required if you want to use the default data collator.
        peft_config (`Dict`, defaults to `None`):
            The PEFT configuration to use for training. If you pass a PEFT configuration, the model will be wrapped in a PEFT model.
    """

    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module] = None,
        ref_model: Union[PreTrainedModel, nn.Module] = None,
        beta: float = 0.1,
        args: TrainingArguments = None,
        data_collator: Optional[DataCollator] = None,
        label_pad_token_id: int = -100,
        padding_value: int = 0,
        truncation_mode: str = "keep_end",
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,        
        student_tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (
            None,
            None,
        ),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        max_length: Optional[int] = None,
        max_prompt_length: Optional[int] = None,
        peft_config: Optional[Dict] = None,
        rec_model_path: str = None,
        teacher_projector = None,
        student_projector = None,
    ):
        self.load_rec_model(rec_model_path)
        self.rec_embed = "SASRec"
        if data_collator is None:
            if tokenizer is None:
                raise ValueError(
                    "max_length or a tokenizer must be specified when using the default DPODataCollatorWithPadding"
                )
            if max_length is None:
                warnings.warn(
                    "When using DPODataCollatorWithPadding, you should set `max_length` in the DPOTrainer's init"
                    " it will be set to `512` by default, but you should do it yourself in the future.",
                    UserWarning,
                )
                max_length = 512
            if max_prompt_length is None:
                warnings.warn(
                    "When using DPODataCollatorWithPadding, you should set `max_prompt_length` in the DPOTrainer's init"
                    " it will be set to `128` by default, but you should do it yourself in the future.",
                    UserWarning,
                )
                max_prompt_length = 128

            data_collator = DPODataCollatorWithPadding(
                tokenizer,
                max_length=max_length,
                max_prompt_length=max_prompt_length,
                label_pad_token_id=label_pad_token_id,
                padding_value=padding_value,
                truncation_mode=truncation_mode,
            )

            if args.remove_unused_columns:
                args.remove_unused_columns = False
                # warn users
                warnings.warn(
                    "When using DPODataCollatorWithPadding, you should set `remove_unused_columns=False` in your TrainingArguments"
                    " we have set it for you, but you should do it yourself in the future.",
                    UserWarning,
                )

            self.use_dpo_data_collator = True
        else:
            self.use_dpo_data_collator = False

        self.label_pad_token_id = label_pad_token_id
        self.padding_value = padding_value

        self.beta = beta
        self.ref_model = ref_model
        self.teacher_llama_tokenizer = tokenizer
        self.student_llama_tokenizer = student_tokenizer
        self.teacher_llama_model = ref_model
        self.student_llama_model = model
        self.student_projector = student_projector
        self.projector = teacher_projector
        self._stored_metrics = defaultdict(lambda: defaultdict(list))

        super().__init__(
            model,
            args,
            data_collator,
            train_dataset,
            eval_dataset,
            tokenizer,
            model_init,
            None,
            callbacks,
            optimizers,
            preprocess_logits_for_metrics,
        )

        # Since we inherit from trainer we always have access to an accelerator
        if hasattr(self, "accelerator"):
            self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)
        else:
            raise AttributeError(
                "Your `Trainer` does not have an `accelerator` object. Consider upgrading `transformers`."
            )
    
    def load_rec_model(self, rec_model_path):
        print('Loading Rec Model')
        self.rec_model = torch.load(rec_model_path, map_location="cpu")

        self.rec_model.eval()
        for name, param in self.rec_model.named_parameters():
            param.requires_grad = False
        print('Loding Rec model Done')
    
    def encode_items(self, seq):
        self.rec_model = self.rec_model.to(seq.device)
        self.projector = self.projector.to(seq.device)
        if self.rec_embed=="SASRec":
            item_rec_embs=self.rec_model.cacu_x(seq)
        elif self.rec_embed in ['Caser','GRU']:
            item_rec_embs=self.rec_model.item_embeddings(seq)
        item_txt_embs=self.projector(item_rec_embs)
        return item_txt_embs
    
    def encode_items_student(self, seq):
        self.rec_model = self.rec_model.to(seq.device)
        self.student_projector = self.student_projector.to(seq.device)
        if self.rec_embed=="SASRec":
            item_rec_embs=self.rec_model.cacu_x(seq)
        elif self.rec_embed in ['Caser','GRU']:
            item_rec_embs=self.rec_model.item_embeddings(seq)
        item_txt_embs=self.student_projector(item_rec_embs)
        return item_txt_embs
    
    def embed_tokens(self, token_ids):
        embeds = self.teacher_llama_model.base_model.embed_tokens(token_ids)
        return embeds

    def wrap_teacher_emb(self, batch):
        input_embeds = self.teacher_llama_model.get_input_embeddings()(batch["tokens"]['input_ids'])
        
        his_token_id=self.teacher_llama_tokenizer("[HistoryEmb]", return_tensors="pt",add_special_tokens=False).input_ids.item()
        cans_token_id=self.teacher_llama_tokenizer("[CansEmb]", return_tensors="pt",add_special_tokens=False).input_ids.item()
        item_token_id=self.teacher_llama_tokenizer("[ItemEmb]", return_tensors="pt",add_special_tokens=False).input_ids.item()
        
        his_item_embeds= self.encode_items(batch["seq"])
        cans_item_embeds= self.encode_items(batch["cans"])
        item_embeds=self.encode_items(batch["item_id"])
            
        for i in range(len(batch["len_seq"])):
            if (batch["tokens"]['input_ids'][i]==his_token_id).nonzero().shape[0]>0:
                idx_tensor=(batch["tokens"]['input_ids'][i]==his_token_id).nonzero().view(-1)
                for idx, item_emb in zip(idx_tensor,his_item_embeds[i,:batch["len_seq"][i].item()]):
                    input_embeds[i,idx]=item_emb
            if (batch["tokens"]['input_ids'][i]==cans_token_id).nonzero().shape[0]>0:
                idx_tensor=(batch["tokens"]['input_ids'][i]==cans_token_id).nonzero().view(-1)
                for idx, item_emb in zip(idx_tensor,cans_item_embeds[i,:batch["len_cans"][i].item()]):
                    input_embeds[i,idx]=item_emb
            if (batch["tokens"]['input_ids'][i]==item_token_id).nonzero().shape[0]>0:
                idx=(batch["tokens"]['input_ids'][i]==item_token_id).nonzero().item()
                input_embeds[i,idx]=item_embeds[i]
        return input_embeds

    def wrap_student_emb(self, batch):
        input_embeds = self.student_llama_model.get_input_embeddings()(batch["tokens"]['input_ids'])
        
        his_token_id=self.student_llama_tokenizer("[HistoryEmb]", return_tensors="pt",add_special_tokens=False).input_ids.item()
        cans_token_id=self.student_llama_tokenizer("[CansEmb]", return_tensors="pt",add_special_tokens=False).input_ids.item()
        item_token_id=self.student_llama_tokenizer("[ItemEmb]", return_tensors="pt",add_special_tokens=False).input_ids.item()
        his_item_embeds= self.encode_items_student(batch["seq"])
        cans_item_embeds= self.encode_items_student(batch["cans"])
        item_embeds=self.encode_items_student(batch["item_id"])
            
        for i in range(len(batch["len_seq"])):
            if (batch["tokens"]['input_ids'][i]==his_token_id).nonzero().shape[0]>0:
                idx_tensor=(batch["tokens"]['input_ids'][i]==his_token_id).nonzero().view(-1)
                for idx, item_emb in zip(idx_tensor,his_item_embeds[i,:batch["len_seq"][i].item()]):
                    input_embeds[i,idx]=item_emb
            if (batch["tokens"]['input_ids'][i]==cans_token_id).nonzero().shape[0]>0:
                idx_tensor=(batch["tokens"]['input_ids'][i]==cans_token_id).nonzero().view(-1)
                for idx, item_emb in zip(idx_tensor,cans_item_embeds[i,:batch["len_cans"][i].item()]):
                    input_embeds[i,idx]=item_emb
            if (batch["tokens"]['input_ids'][i]==item_token_id).nonzero().shape[0]>0:
                idx=(batch["tokens"]['input_ids'][i]==item_token_id).nonzero().item()
                input_embeds[i,idx]=item_embeds[i]
        return input_embeds

    def get_mv_title(self, s):
        sub_list=[", The", ", A", ", An"]
        for sub_s in sub_list:
            if sub_s in s:
                return sub_s[2:]+" "+s.replace(sub_s,"")
        return s
     

    def dpo_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: Dict[str, torch.FloatTensor],
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: Dict[str, torch.FloatTensor],
        reference_free: bool = False,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Compute the DPO loss for a batch of policy and reference model log probabilities.

        Args:
            policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
            policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
            reference_chosen_logps: Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
            reference_rejected_logps: Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)
            beta: Temperature parameter for the DPO loss, typically something in the range of 0.1 to 0.5. We ignore the reference model as beta -> 0.
            reference_free: If True, we ignore the _provided_ reference model and implicitly use a reference model that assigns equal probability to all responses.

        Returns:
            A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
            The losses tensor contains the DPO loss for each example in the batch.
            The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
        """
        # pi_logratios = policy_chosen_logps - policy_rejected_logps
        # for key in policy_rejected_logps:
        # ref_logratios = reference_chosen_logps - reference_rejected_logps
        chosen_logratios = policy_chosen_logps - reference_chosen_logps
        # print(f"chosen:{chosen_logratios}")
        rejected_logratios = {}
        for key in policy_rejected_logps:
            rejected_logratios[key] = policy_rejected_logps[key] - reference_rejected_logps[key]
            # print(f"{key}_logratios:{rejected_logratios[key].shape}")
        # if reference_free:
        #     ref_logratios = 0

        # logits = pi_logratios - ref_logratios
        temp = sum(torch.exp(self.beta * (rejected_logratios[key] - chosen_logratios)) for key in rejected_logratios)
        temp1 = -torch.log(temp)
        losses = -F.logsigmoid(temp1)
        # losses = -F.logsigmoid(self.beta * logits)
        rejected_rewards = {}
        chosen_rewards = self.beta * (policy_chosen_logps - reference_chosen_logps).detach()
        for key in policy_rejected_logps:
            rejected_rewards[key] = self.beta * (policy_rejected_logps[key] - reference_rejected_logps[key]).detach()

        return losses, chosen_rewards, rejected_rewards

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

        # dummy token; we'll ignore the losses on these tokens later
        labels[labels == self.label_pad_token_id] = 0

        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

        if average_log_prob:
            return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
        else:
            return (per_token_logps * loss_mask).sum(-1)

    def teacher_forward(self, batch):
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.

        We do this to avoid doing two forward passes, because it's faster for FSDP.
        """
        concatenated_batch = self.concatenated_inputs(batch)
        # concatenated_batch keys : ['concatenated_input_ids', 'concatenated_attention_mask', 'concatenated_labels']        
        # print(concatenated_batch["concatenated_input_ids"].shape)
        targets = batch["tokens"]['input_ids'].masked_fill(
            batch["tokens"]['input_ids'] == self.teacher_llama_tokenizer.pad_token_id, -100
        ) # [batch_size, max_len]
        # targets = targets.masked_fill((batch["tokens"].token_type_ids == 0)[:,1:], -100)
        targets = targets.masked_fill((batch["tokens"]['token_type_ids'] == 0)[:,:], -100)
        input_embeds = self.wrap_teacher_emb(batch)
        outputs = self.teacher_llama_model(
            inputs_embeds=input_embeds,
            attention_mask=batch["tokens"]["attention_mask"],
            return_dict=True,
            labels=targets,
            output_hidden_states=True,
            use_cache=False
        )
        return outputs 

    
    def student_forward(self, batch):
        concatenated_batch = self.concatenated_inputs(batch)
        # targets = concatenated_batch["tokens"].input_ids.masked_fill(
        #     concatenated_batch["tokens"].input_ids == self.student_llama_tokenizer.pad_token_id, -100) # [batch_size, max_len]
        # # targets = targets.masked_fill((batch["tokens"].token_type_ids == 0)[:,1:], -100)
        # targets = targets.masked_fill((concatenated_batch["tokens"].token_type_ids == 0)[:,:], -100)
        targets = batch["tokens"]['input_ids'].masked_fill(
            batch["tokens"]['input_ids'] == self.student_llama_tokenizer.pad_token_id, -100
        ) # [batch_size, max_len]
        targets = targets.masked_fill((batch["tokens"]['token_type_ids'] == 0)[:,:], -100)
        input_embeds = self.wrap_student_emb(batch)
        outputs = self.student_llama_model(
            inputs_embeds=input_embeds,
            attention_mask=batch["tokens"]["attention_mask"],
            return_dict=True,
            labels=targets,
            output_hidden_states=True,
            use_cache=False
        )
        return outputs 
    
    def concatenated_student_forward(
        self, model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]]
    ) -> Tuple[torch.FloatTensor, Dict[str, torch.FloatTensor], torch.FloatTensor, Dict[str, torch.FloatTensor]]:
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.

        We do this to avoid doing two forward passes, because it's faster for FSDP.
        """
        concatenated_batch = self.concatenated_inputs(batch)
        # concatenated_batch keys : ['concatenated_input_ids', 'concatenated_attention_mask', 'concatenated_labels']        
        # print(concatenated_batch["concatenated_input_ids"].shape)
        all_logits = model(
            concatenated_batch["concatenated_input_ids"],
            attention_mask=concatenated_batch["concatenated_attention_mask"],
        ).logits.to(torch.float32)
        
        targets = batch["tokens"].input_ids.masked_fill(
            batch["tokens"].input_ids == self.teacher_llama_tokenizer.pad_token_id, -100
        ) # [batch_size, max_len]
        # targets = targets.masked_fill((batch["tokens"].token_type_ids == 0)[:,1:], -100)
        targets = targets.masked_fill((batch["tokens"].token_type_ids == 0)[:,:], -100)
        input_embeds = self.wrap_teacher_emb(batch)
        outputs = self.teacher_llama_model(
            inputs_embeds=input_embeds,
            attention_mask=batch["tokens"].attention_mask,
            return_dict=True,
            labels=targets,
            output_hidden_states=True,
            use_cache=False
        )
        
        
        
        all_logps = self._get_batch_logps(
            all_logits,
            concatenated_batch["concatenated_labels"],
            average_log_prob=False,
        )
        chosen_logps = all_logps[: batch["chosen_input_ids"].shape[0]]
        step = batch["chosen_input_ids"].shape[0]
        rejected_logps = {}
        cnt = 0
        for key in batch:
            if key.startswith("rejected") and key.endswith("_input_ids"):
                cnt += 1
                rejected_logps[f"rejected{cnt}"] = all_logps[step*cnt : step*(cnt+1)]

        chosen_logits = all_logits[: batch["chosen_input_ids"].shape[0]]
        rejected_logits = {}
        cnt = 0
        for key in batch:
            if key.startswith("rejected") and key.endswith("_input_ids"):
                cnt += 1
                rejected_logits[f"rejected{cnt}"] = all_logits[step*cnt : step*(cnt+1)]
        return (chosen_logps, rejected_logps, chosen_logits, rejected_logits)

    def concatenated_inputs(self, batch: Dict[str, Union[List, torch.LongTensor]]) -> Dict[str, torch.LongTensor]:
        """Concatenate the chosen and rejected inputs into a single tensor.

        Args:
            batch: A batch of data. Must contain the keys 'chosen_input_ids' and 'rejected_input_ids', which are tensors of shape (batch_size, sequence_length).

        Returns:
            A dictionary containing the concatenated inputs under the key 'concatenated_input_ids'.
        """
        # 把 chosen 和 rejected response 拼接起来
        rejected_max_len = max([batch[key].shape[1] for key in batch if key.startswith("rejected") and key.endswith("_input_ids")])
        max_length = max(batch["chosen_input_ids"].shape[1], rejected_max_len)
        concatenated_batch = {}
        for k in batch:
            if k.startswith("chosen") and isinstance(batch[k], torch.Tensor):
                pad_value = self.label_pad_token_id if "labels" in k else self.padding_value
                concatenated_key = k.replace("chosen", "concatenated")
                concatenated_batch[concatenated_key] = pad_to_length(batch[k], max_length, pad_value=pad_value)
      
        for k in batch:
            if k.startswith("rejected") and isinstance(batch[k], torch.Tensor):
                pad_value = self.label_pad_token_id if "labels" in k else self.padding_value
                # concatenated_key = k.replace("rejected", "concatenated")
                prefix = k.split("_")[0]
                concatenated_key = "concatenated" + k[len(prefix):] 
                concatenated_batch[concatenated_key] = torch.cat(
                    (
                        concatenated_batch[concatenated_key],
                        pad_to_length(batch[k], max_length, pad_value=pad_value),
                    ),
                    dim=0,
                ).to(self.accelerator.device)
        return concatenated_batch

    def calc_hidden_loss(self, teacher_state, student_state, alpha=0.5):
        
        l2_loss = F.mse_loss(student_state, teacher_state)
        cos_loss = 1 - F.cosine_similarity(student_state.view(-1), teacher_state.view(-1), dim=0)    
        total_loss = alpha * l2_loss + (1 - alpha) * cos_loss
        return total_loss
    
    def get_batch_metrics(
        self,
        model,
        batch: Dict[str, Union[List, torch.LongTensor]],
        train_eval: Literal["train", "eval"] = "train",
    ):
        """Compute the DPO loss and other metrics for the given batch of inputs for train or test."""
        metrics = {}
        for name, param in self.projector.named_parameters():
            param.requires_grad = True
        for name, param in self.student_projector.named_parameters():
            param.requires_grad = True
        
        with torch.no_grad():
            out_teacher = self.teacher_forward(batch)

        out_student = self.student_forward(batch)
           
        teacher_hidden_states = [state for i, state in enumerate(out_teacher.hidden_states) if i in [30]]
        student_hidden_states = [state for i, state in enumerate(out_student.hidden_states) if i in [14]]
        distill_loss = 0
        for teacher_state, student_state in zip(teacher_hidden_states, student_hidden_states):
                distill_loss += self.calc_hidden_loss(
                teacher_state, 
                student_state,
                alpha=0.5
            )
        loss = out_student.loss + distill_loss 
        return loss, metrics
        # (
        #     policy_chosen_logps,
        #     policy_rejected_logps,
        #     policy_chosen_logits,
        #     policy_rejected_logits,
        # ) = self.concatenated_forward(model, batch)
        
        # with torch.no_grad():
        #     (
        #         reference_chosen_logps,
        #         reference_rejected_logps,
        #         _,
        #         _,
        #     ) = self.concatenated_forward(self.ref_model, batch)

        # losses, chosen_rewards, rejected_rewards = self.dpo_loss(
        #     policy_chosen_logps,
        #     policy_rejected_logps,
        #     reference_chosen_logps,
        #     reference_rejected_logps,
        # )
        
        # # reward_accuracies 记录 chosen 比所有 rejected 的收益都大的比例是多少
        # reward_accuracies = None
        # for key in rejected_rewards:
        #     if reward_accuracies is None:
        #         reward_accuracies = (chosen_rewards > rejected_rewards[key]).float()
        #     else:
        #         reward_accuracies *= (chosen_rewards > rejected_rewards[key]).float()

        # prefix = "eval_" if train_eval == "eval" else ""
        # metrics[f"{prefix}rewards/chosen"] = chosen_rewards.cpu().numpy().mean()
        # for key in rejected_rewards:
        #     metrics[f"{prefix}rewards/{key}"] = rejected_rewards[key].cpu().numpy().mean()
        # metrics[f"{prefix}rewards/accuracies"] = reward_accuracies.cpu().numpy().mean()
        # for key in rejected_rewards:
        #     metrics[f"{prefix}rewards/margins-{key}"] = (chosen_rewards - rejected_rewards[key]).cpu().numpy().mean()
        # for key in policy_rejected_logps:    
        #     metrics[f"{prefix}logps/rejected-{key}"] = policy_rejected_logps[key].detach().cpu().numpy().mean()
        # metrics[f"{prefix}logps/chosen"] = policy_chosen_logps.detach().cpu().numpy().mean()
        # for key in policy_rejected_logits:    
        #     metrics[f"{prefix}logits/rejected-{key}"] = policy_rejected_logits[key].detach().cpu().numpy().mean()
        # metrics[f"{prefix}logits/chosen"] = policy_chosen_logits.detach().cpu().numpy().mean()

        # return losses.mean(), metrics

    def compute_loss(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs=False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        # print(inputs.keys())
        # print(inputs)
        if not self.use_dpo_data_collator:
            warnings.warn(
                "compute_loss is only implemented for DPODataCollatorWithPadding, and you passed a datacollator that is different than "
                "DPODataCollatorWithPadding - you might see unexpected behavior. Alternatively, you can implement your own prediction_step method if you are using a custom data collator"
            )
        # for student model
        loss, metrics = self.get_batch_metrics(model, inputs, train_eval="train")

        # force log the metrics
        if self.accelerator.is_main_process:
            self.store_metrics(metrics, train_eval="train")

        if return_outputs:
            return (loss, metrics)
        return loss

    def get_batch_samples(self, model, batch: Dict[str, torch.LongTensor]) -> Tuple[str, str]:
        """Generate samples from the model and reference model for the given batch of inputs."""

        policy_output = model.generate(
            batch["prompt_input_ids"],
            attention_mask=batch["prompt_attention_mask"],
            max_length=self.config.max_length,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        reference_output = self.ref_model.generate(
            batch["prompt_input_ids"],
            attention_mask=batch["prompt_attention_mask"],
            max_length=self.config.max_length,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        policy_output = pad_to_length(policy_output, self.config.max_length, self.tokenizer.pad_token_id)
        policy_output_decoded = self.tokenizer.batch_decode(policy_output, skip_special_tokens=True)

        reference_output = pad_to_length(reference_output, self.config.max_length, self.tokenizer.pad_token_id)
        reference_output_decoded = self.tokenizer.batch_decode(reference_output, skip_special_tokens=True)

        return policy_output_decoded, reference_output_decoded

    def prediction_step(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ):
        if not self.use_dpo_data_collator:
            warnings.warn(
                "prediction_step is only implemented for DPODataCollatorWithPadding, and you passed a datacollator that is different than "
                "DPODataCollatorWithPadding - you might see unexpected behavior. Alternatively, you can implement your own prediction_step method if you are using a custom data collator"
            )
        if ignore_keys is None:
            if hasattr(model, "config"):
                ignore_keys = getattr(model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        with torch.no_grad():
            loss, metrics = self.get_batch_metrics(model, inputs, train_eval="eval")

        # force log the metrics
        if self.accelerator.is_main_process:
            self.store_metrics(metrics, train_eval="eval")

        if prediction_loss_only:
            return (loss.detach(), None, None)

        # logits for the chosen and rejected samples from model
        logits_dict = {
            "logits_test/chosen": metrics["logits_test/chosen"],
            # "logits_test/rejected": metrics["logits_test/rejected"],
        }
        logits = tuple(v for k, v in logits_dict.items() if k not in ignore_keys)
        logits = torch.stack(logits).mean(axis=1)
        labels = torch.zeros(logits.shape[0])

        return (loss.detach(), logits, labels)

    def store_metrics(self, metrics: Dict[str, float], train_eval: Literal["train", "eval"] = "train") -> None:
        for key, value in metrics.items():
            self._stored_metrics[train_eval][key].append(value)

    def log(self, logs: Dict[str, float]) -> None:
        """
        Log `logs` on the various objects watching training, including stored metrics.

        Args:
            logs (`Dict[str, float]`):
                The values to log.
        """
        # logs either has 'loss' or 'eval_loss'
        train_eval = "train" if "loss" in logs else "eval"
        # Add averaged stored metrics to logs
        for key, metrics in self._stored_metrics[train_eval].items():
            logs[key] = torch.tensor(metrics).mean().item()
        del self._stored_metrics[train_eval]
        return super().log(logs)

