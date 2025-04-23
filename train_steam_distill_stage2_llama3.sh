export CUDA_VISIBLE_DEVICES=2

python main.py \
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
--llm_path ./llama3_8b \
--rec_model_path ./rec_model/steam.pt \
--ckpt_path /data/cx/checkpoints/steam_llama3/epoch=03-metric=0.473.ckpt \
--ckpt_head_path /data/cx/checkpoints/steam_llama3_distillation_stage1/epoch=04-metric=0.476.ckpt \
--student_pretrained_path ./llama3_1b \
--ckpt_dir ./checkpoints/steam_llama3_distillation_stage2_orth1 \
--output_dir ./output/steam/ \
--log_dir ./log/steam_llama3_stage2_logs \
--lr_warmup_start_lr 5e-6 \
--lr 5e-4 \
--lr_decay_min_lr 5e-6 \
--max_epochs 5