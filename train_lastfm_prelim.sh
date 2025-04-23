export CUDA_VISIBLE_DEVICES=3,4

python main.py \
--mode train \
--batch_size 16 \
--accumulate_grad_batches 2 \
--dataset lastfm_data \
--data_dir ./hf_data/lastfm \
--cans_num 20 \
--prompt_path ./prompt/artist.txt \
--rec_embed SASRec \
--llm_tuning lora \
--llm_path ./hf_model \
--rec_model_path ./rec_model/lastfm.pt \
--ckpt_path ./checkpoints/lastfm_peft/epoch=03-metric=0.541.ckpt \
--ckpt_dir ./checkpoints/lastfm_peft_prelim_layer20/ \
--output_dir ./output/lastfm/ \
--log_dir lastfm_logs \
--lr 7e-4 \
--lr_decay_min_lr 7e-6 \
--max_epochs 15