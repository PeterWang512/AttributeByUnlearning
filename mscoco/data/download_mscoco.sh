mkdir -p data/mscoco

# download mscoco dataset
wget http://images.cocodataset.org/zips/train2017.zip -O data/mscoco/train2017.zip
unzip data/mscoco/train2017.zip -d data/mscoco
rm data/mscoco/train2017.zip

wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip -O data/mscoco/annotations_trainval2017.zip
unzip data/mscoco/annotations_trainval2017.zip -d data/mscoco
rm data/mscoco/annotations_trainval2017.zip

# enable faster huggingface data transfer and download
export HF_HUB_ENABLE_HF_TRANSFER=1

mkdir -p data/mscoco/cache
python -c 'from huggingface_hub import snapshot_download; snapshot_download(repo_id="sywang/AttributeByUnlearning", allow_patterns=["mscoco/*"], repo_type="dataset", local_dir="data", cache_dir="data/mscoco/cache")'

# extract pretrained model weight and fisher information
7z x data/mscoco/model_fisher.7z -odata/mscoco
rm data/mscoco/model_fisher.7z

# extract generated samples
7z x data/mscoco/sample.7z -odata/mscoco
rm data/mscoco/sample.7z

# extract precomputed vae latents, text embeddings
7z x data/mscoco/latents_text_embeddings/latents_text_embeddings.7z.001 -odata/mscoco
rm data/mscoco/latents_text_embeddings/latents_text_embeddings.7z.*
rmdir data/mscoco/latents_text_embeddings

rm -rf data/mscoco/cache
