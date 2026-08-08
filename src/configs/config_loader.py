from pathlib import Path
import yaml


class ConfigLoader:

    _cache = {}

    @classmethod
    def load(cls, relative_path: str):

        if relative_path in cls._cache:
            return cls._cache[relative_path]

        root = Path(__file__).resolve().parents[2]

        file_path = root / "configs" / relative_path

        with open(file_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        cls._cache[relative_path] = config

        return config