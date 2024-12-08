mkdir -p data/abc

# enable faster huggingface data transfer and download
export HF_HUB_ENABLE_HF_TRANSFER=1

mkdir -p data/abc/cache
python -c 'from huggingface_hub import snapshot_download; snapshot_download(repo_id="sywang/GenDataAttribution", allow_patterns=["exemplar/*", "synth_test/*", "laion_subset/*", "models_test/*", "json/test_*", "imagenet_class_to_categories.json", "path_to_prompts.json"], repo_type="dataset", local_dir="data/abc", cache_dir="data/abc/cache")'

python -c 'from huggingface_hub import snapshot_download; snapshot_download(repo_id="sywang/AttributeByUnlearning", allow_patterns=["abc/*"], repo_type="dataset", local_dir="data", cache_dir="data/abc/cache")'

# extract exemplar, synthetic test data, and model weights
7z x data/abc/exemplar/exemplar.7z.001 -odata/abc
rm -rf data/abc/exemplar/exemplar.7z.*

7z x data/abc/synth_test/synth_test.7z.001 -odata/abc
rm -rf data/abc/synth_test/synth_test.7z.*
rmdir data/abc/synth_test

7z x data/abc/laion_subset/laion_subset.7z.001 -odata/abc
rm -rf data/abc/laion_subset/laion_subset.7z.*

7z x data/abc/models_test/models_test.7z.001 -odata/abc
rm -rf data/abc/models_test

# extract precomputed vae latents, text embeddings
7z x data/abc/laion_latents_text_embeddings/laion_latents_text_embeddings.7z.001 -odata/abc
rm -rf data/abc/laion_latents_text_embeddings/laion_latents_text_embeddings.7z.*
rmdir data/abc/laion_latents_text_embeddings

rm -rf data/abc/cache
