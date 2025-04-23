export CUDA_VISIBLE_DEVICES=0
python main_test.py \
--stage distillation_stage2 \
--mode test \
--batch_size 1 \
--accumulate_grad_batches 1 \
--dataset lastfm_data \
--data_dir ./hf_data/lastfm \
--cans_num 20 \
--prompt_path ./prompt/artist.txt \
--rec_embed SASRec \
--llm_tuning lora \
--llm_path ./hf_model \
--rec_model_path ./rec_model/lastfm.pt \
--student_pretrained_path ./tinyllama \
--student_ckpt_path /data/cx/checkpoints/lastfm_llama2_peft_distill_stage2_v2_onlycrosskd/epoch=04-metric=0.492.ckpt \
--output_dir ./output/lastfm_stage2/ \
--log_dir lastfm_logs \
--max_epochs 1


# /data1/chenxiao/LLaRA/checkpoints/lastfm_llama3_peft_distill_stage2_orth/epoch=05-metric=0.557.ckpt