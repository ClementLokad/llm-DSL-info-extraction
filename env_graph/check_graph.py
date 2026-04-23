import json

try:
    with open('data/network/network.json') as f:
        data = json.load(f)

    print("=== 🌟 EXEMPLES DE NOEUDS SCRIPT ===")
    scripts = [n for n in data['nodes'].values() if n['type'] == 'script']
    for s in scripts[10:13]: # Prenons quelques noeuds au hasard
        print(f"🔹 ID: {s['id']}")
        print(f"   Type: {s['type']}")
        print(f"   Name: {s['name']}")

    print("\n=== 🔗 EXEMPLES DE RELATIONS (EDGES) ===")
    for edge_type in ['imports', 'reads', 'writes', 'contains', 'sibling']:
        edges = [e for e in data['edges'] if e['type'] == edge_type]
        if edges:
            print(f"\n[{edge_type.upper()}] ({len(edges)} au total)")
            for e in edges[:3]:
                print(f"  {e['source']} \n    --({edge_type})--> {e['target']}")

except FileNotFoundError:
    print("Le fichier datas/network/network.json n'existe pas. Assurez-vous d'avoir build le graphe.")
