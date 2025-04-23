export CUDA_VISIBLE_DEVICES=2,3

python main.py \
--mode train \
--stage teacher_sft \
--batch_size 1 \
--accumulate_grad_batches 16 \
--dataset lastfm_data \
--data_dir ./hf_data/lastfm \
--cans_num 20 \
--prompt_path ./prompt/artist.txt \
--rec_embed SASRec \
--llm_tuning lora \
--llm_path /home/chen_xiao/Llama-3-70B \
--rec_model_path ./rec_model/lastfm.pt \
--ckpt_dir ./checkpoints/70b-lastfm \
--output_dir ./output/lastfm/ \
--log_dir lastfm_llama2_logs \
--lr_warmup_start_lr 7e-6 \
--lr 7e-4 \
--lr_decay_min_lr 7e-6 \
--max_epochs 5