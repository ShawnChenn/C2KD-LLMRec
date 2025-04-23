python main.py \
--mode train \
--batch_size 8 \
--accumulate_grad_batches 16 \
--dataset movielens_data \
--data_dir hf_dataset/movielens \
--cans_num 20 \
--prompt_path ./prompt/movie.txt \
--rec_embed SASRec \
--llm_tuning lora \
--llm_path hf_llama \
--rec_model_path ./rec_model/movielens.pt \
--ckpt_dir ./checkpoints/movielens/ \
--output_dir ./output/movielens/ \
--log_dir movielens_logs \
--lr_warmup_start_lr 8e-6 \
--lr 8e-4 \
--lr_decay_min_lr 8e-6 \
--max_epochs 5

# nohup sh train_movielens.sh > output.log 2>&1 &
# huggingface-cli download meta-llama/Meta-Llama-3-8B --local-dir Llama-3-8B --repo-type model