export CUDA_VISIBLE_DEVICES=2

python main_test.py \
--mode test \
--stage teacher_sft \
--batch_size 1 \
--accumulate_grad_batches 1 \
--dataset movielens_data \
--data_dir ./hf_data/movielens \
--cans_num 40 \
--prompt_path ./prompt/movie.txt \
--rec_embed SASRec \
--llm_tuning lora \
--llm_path /data1/chenxiao/LLaRA/llama3_8b \
--rec_model_path ./rec_model/movielens.pt \
--ckpt_path /data/cx/checkpoints/movielens_llama3_peft/epoch=04-metric=0.542.ckpt \
--output_dir ./output/movielens-useremb/ \
--log_dir movielens_logs