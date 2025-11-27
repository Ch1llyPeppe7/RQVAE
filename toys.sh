#python train_rqvae.py configs/rqvae_amazon.gin --gin_bind "train.dataset_split='beauty'"
#python train_decoder.py configs/decoder_amazon.gin --gin_bind "train.dataset_split='beauty'"
# python train_rqvae.py configs/rqvae_amazon.gin \
#     --dataset_split 'toys' \
#     --project "Representation Learning" \
#     --runname "RQVAE-toys-entropy"\
#     --save_dir "out/rqvae/amazon/toys/"
python train_decoder.py configs/decoder_amazon.gin \
    --dataset_split 'toys' \
    --project "Representation Learning" \
    --runname "decoder-toys"\
    --save_dir "out/decoder/amazon/toys/"\
    --rqvae_path "out/rqvae/amazon/toys/checkpoint_399999.pt"