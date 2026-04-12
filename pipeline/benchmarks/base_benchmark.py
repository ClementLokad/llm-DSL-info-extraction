from typing import List, Dict, Any

class Benchmark:
    """Classe de base pour les benchmarks."""

    def run(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Méthode à surcharger par les sous-classes."""
        raise NotImplementedError("La méthode 'run' doit être implémentée dans la sous-classe.")
    
    def initialize(self):
        """Méthode d'initialisation optionnelle pour les benchmarks qui en ont besoin."""
        pass