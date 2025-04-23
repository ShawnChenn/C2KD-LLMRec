export CUDA_VISIBLE_DEVICES=4,5,6,7

python main.py \
--mode train \
--batch_size 1 \
--accumulate_grad_batches 32 \
--dataset steam_data \
--data_dir ./hf_data/steam \
--cans_num 20 \
--prompt_path ./prompt/game.txt \
--rec_embed SASRec \
--llm_tuning lora \
--llm_path ./llama3-8b \
--rec_model_path ./rec_model/steam.pt \
--ckpt_dir ./checkpoints/steam_llama3/ \
--output_dir ./output/steam/ \
--log_dir steam_logs \
--lr_warmup_start_lr 5e-6 \
--lr 5e-4 \
--lr_decay_min_lr 5e-6 \
--max_epochs 50