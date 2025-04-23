export CUDA_VISIBLE_DEVICES=0,1

python main.py \
--mode train \
--stage distillation_stage2 \
--batch_size 1 \
--accumulate_grad_batches 16 \
--dataset lastfm_data \
--data_dir ./hf_data/lastfm \
--cans_num 20 \
--prompt_path ./prompt/artist.txt \
--rec_embed SASRec \
--llm_tuning lora \
--llm_path ./hf_model \
--rec_model_path ./rec_model/lastfm.pt \
--ckpt_path /data/cx/checkpoints/lastfm_llama2_peft/epoch=04-metric=0.459.ckpt \
--ckpt_head_path /data/cx/checkpoints/lastfm_llama2_peft_distill_stage1_multilayer/epoch=02-metric=0.475.ckpt \
--student_pretrained_path ./tinyllama \
--student_ckpt_path /data/cx/checkpoints/lastfm_llama2_peft_distill_stage2_rerun_featcos/epoch=04-metric=0.574.ckpt \
--ckpt_dir ./checkpoints/lastfm_llama2_peft_distill_stage2_v2_onlycrosskd/ \
--output_dir ./output/lastfm/ \
--log_dir lastfm_logs \
--lr_warmup_start_lr 4e-6 \
--lr 7e-4 \
--lr_decay_min_lr 7e-6 \
--max_epochs 5