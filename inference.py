from datasets import load_dataset
from transformers import LlamaForCausalLM, LlamaTokenizer, BitsAndBytesConfig, AutoTokenizer
from Prompt import *
import torch
from torch.utils.data import DataLoader
import transformers
from evaluate_batch import evaluate
from peft import PeftModel, prepare_model_for_kbit_training
from accelerate import Accelerator
import fire
from SASRecModules_ori import *
from recommender.A_SASRec_final_bce_llm import SASRec, Caser, GRU
import inspect
import importlib
import mlp_projector
from transformers import AutoTokenizer, TrainingArguments, AutoModelForCausalLM, BitsAndBytesConfig
from peft import get_peft_config, get_peft_model, get_peft_model_state_dict, LoraConfig, TaskType, PeftModel, PeftConfig
from model.model_interface import PeftModelForCausalLM_prelim, PeftModelForCausalLM

def load_rec_model(rec_model_path):
    print('Loading Rec Model')
    rec_model = torch.load(rec_model_path, map_location="cpu")

    rec_model.eval()
    for name, param in rec_model.named_parameters():
        param.requires_grad = False
    return rec_model

def instancialize(Model, rec_size, llm_size):
    class_args = inspect.getargspec(Model.__init__).args[1:]
    args1 = {
        'rec_size': rec_size,
        'llm_size': llm_size
    }
    return Model(**args1)
    
def load_projector(hidden_size):
    projector = instancialize(mlp_projector.MlpProjector, rec_size=64, llm_size=hidden_size)
    for param in projector.parameters():
        param.requires_grad = False
    return projector

def inference( dataset="",
               batch_size: int = 0,
               resume_from_checkpoint: str = "",
               base_model = "",
               prompt_path = "",
               rec_model_path = '',
               ):
    base_model = base_model
    compute_dtype = getattr(torch, "bfloat16")
    # bnb_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_quant_type="nf4",
    #     bnb_4bit_compute_dtype=compute_dtype,
    #     bnb_4bit_use_double_quant=False,
    #     # load_in_8bit=True,
    # )
    device_index = Accelerator().process_index
    device_map = {"": device_index}
    # model = LlamaForCausalLM.from_pretrained(
    #     base_model,
    #     device_map=device_map,
    #     quantization_config=bnb_config,
    # )
    
    if "Llama-3" in base_model:
        tokenizer = AutoTokenizer.from_pretrained(base_model)
    else:
        tokenizer = LlamaTokenizer.from_pretrained(base_model)
        
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.padding_side = "right"
    tokenizer.add_special_tokens({'additional_special_tokens': ['[PH]','[HistoryEmb]','[CansEmb]','[ItemEmb]']})
    
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map=device_map,
    )
    model.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(model, resume_from_checkpoint)
    model.eval()

    projector = load_projector(model.config.hidden_size)
    state_dict = torch.load(os.path.join(resume_from_checkpoint, 'teacher_projector.pt'), map_location="cpu")
    projector.load_state_dict(state_dict)
        
    def convert_dict_to_prompt(d:dict):
        t  = Prompt(prompt_path)
        d["historyList"] = d["historyList"].split("::") if isinstance(d["historyList"], str) else d["historyList"]
        t.historyList = d["historyList"]
        t.itemList = d["itemList"]
        t.trueSelection = d["trueSelection"]
        return t
    
    def generate_and_tokenize_prompt(data_point):
        t = convert_dict_to_prompt(data_point)
        prompt = str(t)
        dic = data_point
        dic["prompt"] = prompt[:-1]
        dic["seq"] = data_point["seq"]
        dic["len_seq"] = data_point["len_seq"]
        dic["item_id"] = data_point["item_id"]
        dic["cans"] = data_point["cans"]            
        dic["len_cans"] = data_point["len_cans"]
        return dic
    

    prompt_path = "prompt.txt" if prompt_path=="" else prompt_path
    # data_files = {
    #     "test": "/data1/chenxiao/S-DPO/sample_data/movielens-sft-cans20/movielens-test.json",
    # }
    data_files = {
        "test": "/data1/chenxiao/S-DPO/sample_data/lastfm-sft-cans20/lastfm-test.json",
    }

    data = load_dataset("json", data_files=data_files)
    data.cleanup_cache_files()
    print(data)

    test_data = data["test"].map(generate_and_tokenize_prompt)
    rec_model = load_rec_model(rec_model_path)

    accuracy, valid_ratio = evaluate(model, tokenizer, test_data, rec_model, projector, batch_size=batch_size)
    print(accuracy, valid_ratio)
    


if __name__ == "__main__":
    fire.Fire(inference)
