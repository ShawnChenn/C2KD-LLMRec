export CUDA_VISIBLE_DEVICES=0,1,2,3

python main.py \
--mode train \
--stage distillation_stage3 \
--batch_size 4 \
--accumulate_grad_batches 16 \
--dataset movielens_data \
--data_dir ./hf_data/movielens \
--cans_num 20 \
--prompt_path ./prompt/movie.txt \
--rec_embed SASRec \
--llm_tuning lora \
--llm_path ./llama3_8b \
--rec_model_path ./rec_model/movielens.pt \
--ckpt_dir ./checkpoints/movielens_llama3_peft_distill_stage3/ \
--ckpt_path /data1/chenxiao/LLaRA/checkpoints/movielens_llama3_peft/epoch=04-metric=0.542.ckpt \
--ckpt_head_path /data1/chenxiao/LLaRA/checkpoints/movielens_llama3_peft_distill_stage1_multilayer/epoch=02-metric=0.667.ckpt \
--student_pretrained_path ./llama3_1b \
--student_ckpt_path /data1/chenxiao/LLaRA/checkpoints/movielens_llama3_peft_distill_stage2_klcosloss/epoch=03-metric=0.500.ckpt \
--output_dir ./output/movielens/ \
--log_dir movielens_logs \
--lr_warmup_start_lr 8e-6 \
--lr 4e-5 \
--lr_decay_min_lr 4e-6 \
--max_epochs 3