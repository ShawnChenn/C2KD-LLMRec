export CUDA_VISIBLE_DEVICES=0,1

python main.py \
--mode train \
--stage distillation_stage1 \
--batch_size 8 \
--accumulate_grad_batches 16 \
--dataset movielens_data \
--data_dir ./hf_data/movielens \
--cans_num 20 \
--prompt_path ./prompt/movie.txt \
--rec_embed SASRec \
--llm_tuning lora \
--llm_path ./hf_model \
--rec_model_path ./rec_model/movielens.pt \
--ckpt_path /data1/chenxiao/LLaRA/checkpoints/movielens_llama2_peft/epoch=04-metric=0.532.ckpt \
--ckpt_dir ./checkpoints/movielens_llama2_peft_distill_stage1_multilayer/ \
--output_dir ./output/movielens/ \
--log_dir movielens_logs \
--lr_warmup_start_lr 8e-6 \
--lr 8e-4 \
--lr_decay_min_lr 8e-6 \
--max_epochs 3