
python merge_property.py --datapath ./physxnet   # your physxnet path

python gen_csv.py

python retrieval_texture_example.py

python build_metadata.py PhysXNet --output_dir ../datasets/PhysXNet

python render.py PhysXNet --output_dir ../datasets/PhysXNet

python build_metadata.py PhysXNet --output_dir ../datasets/PhysXNet
python render_cond.py PhysXNet --output_dir ../datasets/PhysXNet

python build_metadata.py PhysXNet --output_dir ../datasets/PhysXNet

python voxelize.py PhysXNet --output_dir ../datasets/PhysXNet
python build_metadata.py PhysXNet --output_dir ../datasets/PhysXNet

python extract_feature.py --output_dir ../datasets/PhysXNet
python build_metadata.py PhysXNet --output_dir ../datasets/PhysXNet


python encode_latent.py --output_dir ../datasets/PhysXNet
python encode_latent_phy.py --output_dir ../datasets/PhysXNet
python build_metadata.py PhysXNet --output_dir ../datasets/PhysXNet



