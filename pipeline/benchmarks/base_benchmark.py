from typing import List, Dict, Any

class Benchmark:
    """Classe de base pour les benchmarks."""

    def run(self, data: List[Dict[str, Any]]) -> Any:
        """Méthode à surcharger par les sous-classes."""
        raise NotImplementedError("La méthode 'run' doit être implémentée dans la sous-classe.")