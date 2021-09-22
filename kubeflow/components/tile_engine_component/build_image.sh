#!/bin/sh

image_name=asia.gcr.io/scancer/ae13/kubeflow/tile_engine
image_tag=latest

full_image_name=${image_name}:${image_tag}
base_image_tag=tf2.4

cd ../.. 

docker build -f ./components/tile_engine_component/Dockerfile --build-arg BASE_IMAGE_TAG=${base_image_tag} -t "${full_image_name}" .
docker push "$full_image_name"