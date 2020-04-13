#!/usr/bin/env sh
# This scripts downloads the pre-trained models.

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
cd $DIR

echo "Downloading pre-trained models..."
wget https://github.com/kivantium/illustration2vec/releases/download/v2.1.0/tag_list.json.gz
wget https://github.com/kivantium/illustration2vec/releases/download/v2.1.0/illust2vec_tag_ver200.onnx
wget https://github.com/kivantium/illustration2vec/releases/download/v2.1.0/illust2vec_ver200.onnx
gunzip tag_list.json.gz

echo "Done."
