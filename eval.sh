#python train_rqvae.py configs/rqvae_amazon.gin --gin_bind "train.dataset_split='beauty'"
#python train_decoder.py configs/decoder_amazon.gin --gin_bind "train.dataset_split='beauty'"
python eval_rqvae.py configs/rqvae_amazon_eval.gin \
    --dataset_split 'toys' \
    --project "Representation Learning" \
    --runname "RQVAE-toys-eval"\
    --save_dir "out/rqvae/amazon/beauty/"\
    --rqvae_path "out/rqvae/amazon/toys/checkpoint_399999.pt"
    --noise_test 'True'