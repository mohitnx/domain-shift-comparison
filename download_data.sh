#!/bin/bash
# downloads and arranges the datasets both scripts expect.
# run this once from the repo root before running either project script.
set -e

mkdir -p data

# --- project 1: plantvillage (source domain) + plantdoc (target domain) ---
if [ ! -d data/plantvillage_tomato ]; then
  git clone --depth 1 https://github.com/spMohanty/PlantVillage-Dataset.git /tmp/pv
  mkdir -p data/plantvillage_tomato
  for d in /tmp/pv/raw/color/Tomato*; do
    cp -r "$d" "data/plantvillage_tomato/$(basename "$d")"
  done
  rm -rf /tmp/pv
fi

if [ ! -d data/plantdoc_tomato ]; then
  git clone --depth 1 https://github.com/pratikkayal/PlantDoc-Dataset.git /tmp/pd
  mkdir -p data/plantdoc_tomato/train data/plantdoc_tomato/test
  classes=(
    "Tomato Early blight leaf" "Tomato Septoria leaf spot" "Tomato leaf"
    "Tomato leaf bacterial spot" "Tomato leaf late blight"
    "Tomato leaf mosaic virus" "Tomato leaf yellow virus" "Tomato mold leaf"
  )
  for c in "${classes[@]}"; do
    cp -r "/tmp/pd/train/$c" "data/plantdoc_tomato/train/" 2>/dev/null || true
    cp -r "/tmp/pd/test/$c" "data/plantdoc_tomato/test/" 2>/dev/null || true
  done
  rm -rf /tmp/pd
fi

# --- project 2: coffee leaf severity dataset ---
if [ ! -d lara2018 ]; then
  git clone --depth 1 https://github.com/esgario/lara2018.git
  rm -rf lara2018/.git lara2018/segmentation
fi

echo "datasets ready: data/plantvillage_tomato, data/plantdoc_tomato, lara2018/"
