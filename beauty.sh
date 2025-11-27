#python train_rqvae.py configs/rqvae_amazon.gin --gin_bind "train.dataset_split='beauty'"
#python train_decoder.py configs/decoder_amazon.gin --gin_bind "train.dataset_split='beauty'"
# python train_rqvae.py configs/rqvae_amazon.gin \
#     --dataset_split 'beauty' \
#     --project "Representation Learning" \
#     --runname "RQVAE-beauty-norm"\
#     --save_dir "out/rqvae/amazon/beauty/norm_reg/"
python train_decoder.py configs/decoder_amazon.gin \
    --dataset_split 'beauty' \
    --project "Representation Learning" \
    --runname "decoder-beauty-norm-69999"\
    --save_dir "out/decoder/amazon/beauty/norm_reg/"\
    --rqvae_path "out/rqvae/amazon/beauty/norm_regcheckpoint_69999.pt"