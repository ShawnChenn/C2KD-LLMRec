export CUDA_VISIBLE_DEVICES=0,1,2,3

python main.py \
--mode train \
--stage distillation_stage_prelim \
--batch_size 8 \
--accumulate_grad_batches 2 \
--dataset movielens_data \
--data_dir ./hf_data/movielens \
--cans_num 20 \
--prompt_path ./prompt/movie.txt \
--rec_embed SASRec \
--llm_tuning lora \
--llm_path ./llama3_8b \
--rec_model_path ./rec_model/movielens.pt \
--ckpt_dir /data/cx/llara_ckpt/movielens_prelim10/ \
--output_dir ./output/movielens/ \
--log_dir movielens_logs \
--lr_warmup_start_lr 8e-6 \
--lr 8e-4 \
--lr_decay_min_lr 8e-6 \
--max_epochs 5