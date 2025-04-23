export CUDA_VISIBLE_DEVICES=2

python main_test.py \
--stage teacher_sft \
--mode test \
--batch_size 1 \
--accumulate_grad_batches 1 \
--dataset lastfm_data \
--data_dir ./hf_data/lastfm \
--cans_num 20 \
--prompt_path ./prompt/artist.txt \
--rec_embed SASRec \
--llm_tuning lora \
--llm_path ./llama3_1b \
--rec_model_path ./rec_model/lastfm.pt \
--ckpt_path  /data1/chenxiao/LLaRA/checkpoints/lastfm_llama3_peft_distill_stage2_orth0218/epoch=03-metric=0.492.ckpt \
--output_dir ./output/lastfm/ 

# 
# /data/cx/checkpoints/lastfm_llama3_1b_peft/epoch=04-metric=0.426.ckpt 