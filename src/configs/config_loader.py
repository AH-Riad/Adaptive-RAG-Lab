from pathlib import Path
import yaml

class ConfigLoader:
    _cache = {}

    @classmethod
    def load(cls, relative_path: str):
        if relative_path in cls._cache:
            return cls._cache[relative_path]

        # Get the directory where this config_loader.py file is located (src/configs)
        base_dir = Path(__file__).parent.resolve()
        
        # Combine it with the relative path (e.g., planning/retrieval_policy.yaml)
        file_path = base_dir / relative_path

        with open(file_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        cls._cache[relative_path] = config
        return config