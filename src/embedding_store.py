from copy import deepcopy
import os

import numpy as np
import pandas as pd

from src.utils import compute_mdhash_id


class EmbeddingStore:
    def __init__(self, embedding_model, db_filename, batch_size, namespace):
        self.embedding_model = embedding_model
        self.db_filename = db_filename
        self.batch_size = batch_size
        self.namespace = namespace

        self.hash_ids = []
        self.texts = []
        self.embeddings = []
        self.hash_id_to_text = {}
        self.hash_id_to_idx = {}
        self.text_to_hash_id = {}

        self._load_data()

    def _load_data(self):
        if os.path.exists(self.db_filename):
            df = pd.read_parquet(self.db_filename)
            self.hash_ids = df["hash_id"].values.tolist()
            self.texts = df["text"].values.tolist()
            self.embeddings = df["embedding"].values.tolist()

            self.hash_id_to_idx = {h: idx for idx, h in enumerate(self.hash_ids)}
            self.hash_id_to_text = {h: t for h, t in zip(self.hash_ids, self.texts)}
            self.text_to_hash_id = {t: h for t, h in zip(self.texts, self.hash_ids)}
            print(f"[{self.namespace}] Loaded {len(self.hash_ids)} records from {self.db_filename}")

    def insert_text(self, text_list, item_ids=None):
        if item_ids is not None and len(item_ids) != len(text_list):
            raise ValueError("item_ids length must match text_list length")

        nodes_dict = {}
        for idx, text in enumerate(text_list):
            stable_key = item_ids[idx] if item_ids is not None else text
            hash_id = compute_mdhash_id(str(stable_key), prefix=self.namespace + "-")
            nodes_dict[hash_id] = {"content": text}

        all_hash_ids = list(nodes_dict.keys())
        existing = set(self.hash_ids)
        missing_ids = [h for h in all_hash_ids if h not in existing]
        texts_to_encode = [nodes_dict[h]["content"] for h in missing_ids]

        if texts_to_encode:
            all_embeddings = self.embedding_model.encode(
                texts_to_encode,
                normalize_embeddings=True,
                show_progress_bar=False,
                batch_size=self.batch_size,
            )
            self._upsert(missing_ids, texts_to_encode, all_embeddings)

    def insert_image(self, image_paths, item_ids=None):
        from PIL import Image

        if item_ids is not None and len(item_ids) != len(image_paths):
            raise ValueError("item_ids length must match image_paths length")

        nodes_dict = {}
        for idx, path in enumerate(image_paths):
            stable_key = item_ids[idx] if item_ids is not None else path
            hash_id = compute_mdhash_id(str(stable_key), prefix=self.namespace + "-")
            nodes_dict[hash_id] = {"path": path}

        all_hash_ids = list(nodes_dict.keys())
        existing = set(self.hash_ids)
        missing_ids = [h for h in all_hash_ids if h not in existing]

        if missing_ids:
            images_to_encode = []
            valid_ids = []
            valid_paths = []
            for h in missing_ids:
                path = nodes_dict[h]["path"]
                try:
                    img = Image.open(path).convert("RGB")
                    images_to_encode.append(img)
                    valid_ids.append(h)
                    valid_paths.append(path)
                except Exception as e:
                    print(f"failed to load image {path}: {e}")

            if images_to_encode:
                all_embeddings = self.embedding_model.encode(images_to_encode, normalize_embeddings=True)
                self._upsert(valid_ids, valid_paths, all_embeddings)

    def _upsert(self, hash_ids, texts, embeddings):
        self.hash_ids.extend(hash_ids)
        self.texts.extend(texts)
        self.embeddings.extend(embeddings)

        self.hash_id_to_idx = {h: idx for idx, h in enumerate(self.hash_ids)}
        self.hash_id_to_text = {h: t for h, t in zip(self.hash_ids, self.texts)}
        self.text_to_hash_id = {t: h for t, h in zip(self.texts, self.hash_ids)}
        self._save_data()

    def _save_data(self):
        df = pd.DataFrame(
            {
                "hash_id": self.hash_ids,
                "text": self.texts,
                "embedding": self.embeddings,
            }
        )
        os.makedirs(os.path.dirname(self.db_filename), exist_ok=True)
        df.to_parquet(self.db_filename, index=False)

    def get_hash_id_to_text(self):
        return deepcopy(self.hash_id_to_text)

    def encode_texts(self, texts):
        return self.embedding_model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False, batch_size=self.batch_size
        )

    def get_embeddings(self, hash_ids):
        if not hash_ids:
            return np.array([])
        indices = np.array([self.hash_id_to_idx[h] for h in hash_ids], dtype=np.intp)
        return np.array(self.embeddings)[indices]

