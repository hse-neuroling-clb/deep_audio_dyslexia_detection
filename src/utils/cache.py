import pickle
import os


class Cache:
    def __init__(self, prekey: str):
        self._cache_folder = f"./data/cache"
        if not os.path.exists(self._cache_folder):
            os.makedirs(self._cache_folder)

        self._cache_path = os.path.join(self._cache_folder, f"{prekey}.pkl")
        if os.path.exists(self._cache_path):
            with open(self._cache_path, "rb") as f:
                self.cache = pickle.load(f)
        else:
            self.cache = {}

    def __getitem__(self, item):
        return self.cache[item]

    def __len__(self):
        return len(self.cache)

    def items(self):
        return self.cache.items()

    def __contains__(self, item):
        return item in self.cache

    def __setitem__(self, key, value):
        self.cache[key] = value
        self._save()
    
    def _save(self):
        with open(self._cache_path, "wb") as f:
            pickle.dump(self.cache, f, protocol=pickle.HIGHEST_PROTOCOL)