export CUDA_VISIBLE_DEVICES=0

python main_test.py \
--mode test \
--stage teacher_sft \
--batch_size 1 \
--accumulate_grad_batches 1 \
--dataset steam_data \
--data_dir ./hf_data/steam \
--cans_num 40 \
--prompt_path ./prompt/game.txt \
--rec_embed SASRec \
--llm_tuning lora \
--llm_path ./hf_model \
--rec_model_path ./rec_model/steam.pt \
--ckpt_path /data/cx/checkpoints/steam_llama2_peft/epoch=04-metric=0.318.ckpt \
--output_dir ./output/steam/ \
--log_dir steam_logs \
--lr_warmup_start_lr 5e-6 \
--lr 5e-4 \
--lr_decay_min_lr 5e-6 \
--max_epochs 1