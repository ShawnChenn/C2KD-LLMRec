export CUDA_VISIBLE_DEVICES=0,1,2,3

torchrun --nproc_per_node 4 --master_port=24742 main.py \
--mode train \
--stage distillation_stage2 \
--batch_size 8 \
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
--ckpt_head_path /data1/chenxiao/LLaRA/checkpoints/steam_llama2_distillation_stage1/epoch=02-metric=0.470.ckpt \
--student_pretrained_path ./tinyllama \
--ckpt_dir ./checkpoints/steam_llama2_distillation_stage2_klcosloss/ \
--output_dir ./output/steam/ \
--log_dir steam_logs \
--lr_warmup_start_lr 5e-6 \
--lr 5e-4 \
--lr_decay_min_lr 5e-6 \
--max_epochs 6