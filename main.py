#!/usr/bin/env python3

import os
import sys
from PIL import Image
import torch
import clip
import argparse
import time
from datetime import datetime
from collections import defaultdict
from pathlib import Path
import numpy as np
from tqdm import tqdm  # progress bar

# Bright colors
RED = "\033[1;31m"
GREEN = "\033[1;32m"
YELLOW = "\033[1;33m"
CYAN = "\033[1;36m"
MAGENTA = "\033[1;35m"
RESET = "\033[0m"

def human_readable_size(size, decimal_places=2):
    for unit in ['B','KB','MB','GB','TB']:
        if size < 1024:
            return f"{size:.{decimal_places}f} {unit}"
        size /= 1024
    return f"{size:.{decimal_places}f} PB"

def get_all_images(root_dir):
    exts = ('.jpg','.jpeg','.png','.bmp','.gif','.tiff','.webp')
    return [str(p) for p in Path(root_dir).rglob('*') if p.suffix.lower() in exts]

def get_image_resolution(image_path):
    try:
        with Image.open(image_path) as img:
            return img.width * img.height
    except:
        return 0

def get_file_size(image_path):
    try:
        return os.path.getsize(image_path)
    except:
        return 0

def load_clip_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess, device

def compute_embeddings(images, model, preprocess, device):
    embeddings = {}
    print(f"{MAGENTA}Computing embeddings for {len(images)} images...{RESET}")
    for img_path in tqdm(images, desc=f"{CYAN}Processing Images{RESET}", ncols=100):
        try:
            image = preprocess(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)
            with torch.no_grad():
                emb = model.encode_image(image)
                emb /= emb.norm(dim=-1, keepdim=True)
                embeddings[img_path] = emb.cpu().numpy()[0]
        except Exception as e:
            print(f"{RED}Failed to process {img_path}: {e}{RESET}")
    return embeddings

def find_similar_images(embeddings, similarity_threshold=0.95):
    used = set()
    duplicates = defaultdict(list)
    paths = list(embeddings.keys())
    emb_array = np.array([embeddings[p] for p in paths])
    
    print(f"{MAGENTA}Comparing images for similarity...{RESET}")
    for i, p1 in tqdm(enumerate(paths), total=len(paths), desc=f"{CYAN}Finding Duplicates{RESET}", ncols=100):
        if p1 in used:
            continue
        sim_group = [p1]
        for j in range(i+1, len(paths)):
            p2 = paths[j]
            if p2 in used:
                continue
            sim = np.dot(emb_array[i], emb_array[j])
            if sim >= similarity_threshold:
                sim_group.append(p2)
                used.add(p2)
        if len(sim_group) > 1:
            duplicates[p1] = sim_group[1:]  # keep p1 as representative
    return duplicates

def keep_best_and_delete(duplicates):
    total_deleted = 0
    total_space_saved = 0
    for rep, dup_list in duplicates.items():
        all_files = [rep] + dup_list
        files_res = [(f, get_image_resolution(f), get_file_size(f)) for f in all_files]
        files_res.sort(key=lambda x: x[1], reverse=True)
        best_file = files_res[0][0]
        for f, _, size in files_res[1:]:
            try:
                os.remove(f)
                total_deleted += 1
                total_space_saved += size
                print(f"{YELLOW}Deleted duplicate: {f} ({human_readable_size(size)}){RESET}")
            except Exception as e:
                print(f"{RED}Failed to delete {f}: {e}{RESET}")
        print(f"{GREEN}Kept: {best_file}{RESET}")
    return total_deleted, total_space_saved

def main():
    parser = argparse.ArgumentParser(description="AI-based Duplicate Image Cleaner (CLIP version)")
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    parser.add_argument('-t', '--threshold', type=float, default=0.95, help='Similarity threshold (0-1)')
    args = parser.parse_args()

    dir_path = input(f"{CYAN}Enter directory containing images: {RESET}").strip()
    if not os.path.isdir(dir_path):
        print(f"{RED}Directory not found!{RESET}")
        sys.exit(1)

    start_time = time.time()
    print(f"{MAGENTA}Scanning for images...{RESET}")
    images = get_all_images(dir_path)
    total_images = len(images)
    print(f"{GREEN}Found {total_images} images.{RESET}")

    if total_images == 0:
        sys.exit(0)

    print(f"{MAGENTA}Loading CLIP model...{RESET}")
    model, preprocess, device = load_clip_model()

    embeddings = compute_embeddings(images, model, preprocess, device)
    duplicates = find_similar_images(embeddings, similarity_threshold=args.threshold)

    if duplicates:
        print(f"{MAGENTA}Processing duplicates...{RESET}")
        deleted_count, space_saved = keep_best_and_delete(duplicates)
    else:
        print(f"{GREEN}No duplicates found.{RESET}")
        deleted_count, space_saved = 0, 0

    end_time = time.time()
    elapsed_time = end_time - start_time
    end_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print(f"\n{CYAN}===== Summary ====={RESET}")
    print(f"{GREEN}Total images scanned: {total_images}{RESET}")
    print(f"{YELLOW}Duplicates removed: {deleted_count}{RESET}")
    print(f"{MAGENTA}Total space saved: {human_readable_size(space_saved)}{RESET}")
    print(f"{CYAN}Total time taken: {elapsed_time:.2f} seconds{RESET}")
    print(f"{CYAN}Job ended at: {end_datetime}{RESET}")

if __name__ == "__main__":
    main()
