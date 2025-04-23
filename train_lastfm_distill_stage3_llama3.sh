export CUDA_VISIBLE_DEVICES=0,1,2,3

python  main.py \
--mode train \
--stage distillation_stage3 \
--batch_size 4 \
--accumulate_grad_batches 16 \
--dataset lastfm_data \
--data_dir ./hf_data/lastfm \
--cans_num 20 \
--prompt_path ./prompt/artist.txt \
--rec_embed SASRec \
--llm_tuning lora \
--llm_path ./llama3_8b \
--rec_model_path ./rec_model/lastfm.pt \
--ckpt_path /data1/chenxiao/LLaRA/checkpoints/lastfm_llama3_8b_peft/epoch=03-metric=0.475.ckpt \
--ckpt_head_path /data1/chenxiao/LLaRA/checkpoints/lastfm_llama3_peft_distill_stage1_multilayer/epoch=02-metric=0.426.ckpt \
--student_pretrained_path ./llama3_1b \
--student_ckpt_path /data1/chenxiao/LLaRA/checkpoints/lastfm_llama3_peft_distill_stage2_top1cosmoe_adamw_klcosloss/epoch=07-metric=0.623.ckpt \
--ckpt_dir ./checkpoints/lastfm_llama3_peft_distill_stage3/ \
--output_dir ./output/lastfm/ \
--log_dir lastfm_logs \
--lr_warmup_start_lr 7e-6 \
--lr 1e-4 \
--lr_decay_min_lr 7e-6 \
--max_epochs 3