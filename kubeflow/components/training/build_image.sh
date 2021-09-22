#!/bin/sh

image_name=asia.gcr.io/scancer/ae13/kubeflow/training
image_tag=latest

full_image_name=${image_name}:${image_tag}
base_image_tag=tf2.4

cd "$(dirname "$0")" 

docker build --build-arg BASE_IMAGE_TAG=${base_image_tag} -t "${full_image_name}" .
docker push "$full_image_name"