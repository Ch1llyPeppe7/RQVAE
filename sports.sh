#python train_rqvae.py configs/rqvae_amazon.gin --gin_bind "train.dataset_split='beauty'"
#python train_decoder.py configs/decoder_amazon.gin --gin_bind "train.dataset_split='beauty'"
python train_rqvae.py configs/rqvae_amazon.gin \
    --dataset_split 'sports' \
    --project "Representation Learning" \
    --runname "RQVAE-sports-entropy"\
    --save_dir "out/rqvae/amazon/sports/"
#python train_decoder.py configs/decoder_amazon.gin --gin_bind "train.dataset_split='toys'" --gin_bind "train.runname='RQVAE-decoder-toys'"