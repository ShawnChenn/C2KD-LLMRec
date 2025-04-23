export CUDA_VISIBLE_DEVICES=0,1,2,3

python main.py \
--mode train \
--stage distillation_stage1 \
--batch_size 4 \
--accumulate_grad_batches 16 \
--dataset steam_data \
--data_dir ./hf_data/steam \
--cans_num 20 \
--prompt_path ./prompt/game.txt \
--rec_embed SASRec \
--llm_tuning lora \
--llm_path ./hf_model \
--rec_model_path ./rec_model/steam.pt \
--ckpt_path /data1/chenxiao/LLaRA/checkpoints/steam_llama2_peft/epoch=04-metric=0.318.ckpt \
--ckpt_dir ./checkpoints/steam_llama2_distillation_stage1/ \
--output_dir ./output/steam/ \
--log_dir steam_llama3_distillation_stage1_logs \
--lr_warmup_start_lr 5e-6 \
--lr 5e-4 \
--lr_decay_min_lr 5e-6 \
--max_epochs 3