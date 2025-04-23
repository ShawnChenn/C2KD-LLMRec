export CUDA_VISIBLE_DEVICES=3

python main_test.py \
--stage distillation_stage2 \
--mode test \
--batch_size 1 \
--accumulate_grad_batches 1 \
--dataset movielens_data \
--data_dir ./hf_data/movielens \
--cans_num 40 \
--prompt_path ./prompt/movie.txt \
--rec_embed SASRec \
--llm_tuning lora \
--llm_path ./llama3_8b \
--rec_model_path ./rec_model/movielens.pt \
--student_pretrained_path ./llama3_1b \
--student_ckpt_path /data/cx/checkpoints/movielens_llama3_peft_distill_stage2_crosskdv2/epoch=02-metric=0.426.ckpt \
--output_dir ./output/movielens/ \
--log_dir movielens_logs 