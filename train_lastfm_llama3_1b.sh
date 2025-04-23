export CUDA_VISIBLE_DEVICES=0,1,2,3

python main.py \
--mode train \
--stage teacher_sft \
--batch_size 8 \
--accumulate_grad_batches 16 \
--dataset lastfm_data \
--data_dir ./hf_data/lastfm \
--cans_num 20 \
--prompt_path ./prompt/artist.txt \
--rec_embed SASRec \
--llm_tuning lora \
--llm_path ./llama3_1b \
--rec_model_path ./rec_model/lastfm.pt \
--ckpt_dir ./checkpoints/lastfm_llama3_1b_peft/ \
--output_dir ./output/lastfm/ \
--log_dir lastfm_tinyllama_logs \
--lr_warmup_start_lr 7e-6 \
--lr 4e-4 \
--lr_decay_min_lr 4e-6 \
--max_epochs 5