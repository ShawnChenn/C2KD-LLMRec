export CUDA_VISIBLE_DEVICES=0,1

python main.py \
--mode train \
--stage distillation_stage1 \
--batch_size 8 \
--accumulate_grad_batches 16 \
--dataset lastfm_data \
--data_dir ./hf_data/lastfm \
--cans_num 20 \
--prompt_path ./prompt/artist.txt \
--rec_embed SASRec \
--llm_tuning lora \
--llm_path /data/cx/opt-6.7b \
--rec_model_path ./rec_model/lastfm.pt \
--ckpt_path /data/cx/llara_ckpt/opt7b/epoch=00-metric=0.230.ckpt \
--ckpt_dir /data/cx/llara_ckpt/lastfm_opt7b_peft_distill_stage1_multilayer/ \
--output_dir ./output/lastfm/ \
--log_dir lastfm_llama3_distill_stage1_logs \
--lr_warmup_start_lr 7e-6 \
--lr 7e-4 \
--lr_decay_min_lr 7e-6 \
--max_epochs 3