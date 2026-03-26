# Expanded NYC Synthetic PAS-RAG Dataset

- Anchors: 30
- Chunks: 1010
- Queries: 423

## Category counts
- bar: 40
- bookstore: 55
- cafe: 110
- grocery_store: 80
- hospital: 25
- hotel: 88
- library: 45
- museum: 55
- park: 95
- pharmacy: 70
- restaurant: 183
- subway_station: 124
- theater: 40

## Query complexity counts
- complex: 110
- intermediate: 150
- seed: 3
- simple: 160

## Generation approach
Synthetic POIs were sampled around neighborhood centers across all five boroughs, enriched with category-specific metadata, tagged against the nearest PAS anchors, and used to derive evaluation queries with simple, intermediate, and complex semantic filters.
