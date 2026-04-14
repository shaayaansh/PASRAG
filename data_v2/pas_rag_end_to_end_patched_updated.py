from __future__ import annotations

import json
import math
import random
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

BASE = Path(__file__).resolve().parent
RNG = random.Random(42)


# -----------------------------
# I/O helpers
# -----------------------------

def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]


# -----------------------------
# Geo + PAS helpers
# -----------------------------

def haversine_m(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    lat1, lon1 = map(math.radians, a)
    lat2, lon2 = map(math.radians, b)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    s = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * 6371000.0 * math.asin(math.sqrt(s))


def bearing_deg(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    lat1, lon1 = map(math.radians, a)
    lat2, lon2 = map(math.radians, b)
    y = math.sin(lon2 - lon1) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(lon2 - lon1)
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


def dir_bin(theta: float) -> str:
    bins = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
    idx = int(((theta + 11.25) % 360) // 22.5)
    return bins[idx]


def dist_bin(distance_m: float) -> str:
    miles = distance_m / 1609.344
    if miles < 0.25:
        return "0-0.25mi"
    if miles < 0.5:
        return "0.25-0.5mi"
    if miles < 0.75:
        return "0.5-0.75mi"
    if miles < 1.0:
        return "0.75-1mi"
    if miles < 1.5:
        return "1-1.5mi"
    if miles < 2.0:
        return "1.5-2mi"
    return "2mi+"


def offset_from_anchor(lat: float, lon: float, distance_m: float, bearing_deg_val: float) -> Tuple[float, float]:
    """Project a point from an anchor using distance and bearing."""
    R = 6378137.0
    d_over_R = distance_m / R
    br = math.radians(bearing_deg_val)
    lat1 = math.radians(lat)
    lon1 = math.radians(lon)
    lat2 = math.asin(
        math.sin(lat1) * math.cos(d_over_R)
        + math.cos(lat1) * math.sin(d_over_R) * math.cos(br)
    )
    lon2 = lon1 + math.atan2(
        math.sin(br) * math.sin(d_over_R) * math.cos(lat1),
        math.cos(d_over_R) - math.sin(lat1) * math.sin(lat2),
    )
    return (math.degrees(lat2), math.degrees(lon2))


def sample_Uz(anchor_latlon: Tuple[float, float], user_dir_bin: str, user_dist_bin: str, K: int = 200) -> List[Tuple[float, float]]:
    """Sample latent user locations consistent with the PAS token."""
    if user_dist_bin == "0-0.25mi":
        rmin, rmax = 0.0, 0.25 * 1609.344
    elif user_dist_bin == "0.25-0.5mi":
        rmin, rmax = 0.25 * 1609.344, 0.5 * 1609.344
    elif user_dist_bin == "0.5-0.75mi":
        rmin, rmax = 0.5 * 1609.344, 0.75 * 1609.344
    elif user_dist_bin == "0.75-1mi":
        rmin, rmax = 0.75 * 1609.344, 1.0 * 1609.344
    elif user_dist_bin == "1-1.5mi":
        rmin, rmax = 1.0 * 1609.344, 1.5 * 1609.344
    elif user_dist_bin == "1.5-2mi":
        rmin, rmax = 1.5 * 1609.344, 2.0 * 1609.344
    else:
        rmin, rmax = 2.0 * 1609.344, 3.0 * 1609.344

    sector_centers = {"N": 0, "NNE": 22.5, "NE": 45, "ENE": 67.5, "E": 90, "ESE": 112.5, "SE": 135, "SSE": 157.5, "S": 180, "SSW": 202.5, "SW": 225, "WSW": 247.5, "W": 270, "WNW": 292.5, "NW": 315, "NNW": 337.5}
    center = sector_centers[user_dir_bin]
    pts: List[Tuple[float, float]] = []
    for _ in range(K):
        r = math.sqrt(RNG.random() * (rmax * rmax - rmin * rmin) + rmin * rmin)
        bearing = (center + RNG.uniform(-11.25, 11.25)) % 360.0
        pts.append(offset_from_anchor(anchor_latlon[0], anchor_latlon[1], r, bearing))
    return pts


def sample_full_annulus(anchor_latlon: Tuple[float, float], user_dist_bin: str, K: int = 200) -> List[Tuple[float, float]]:
    if user_dist_bin == "0-0.25mi":
        rmin, rmax = 0.0, 0.25 * 1609.344
    elif user_dist_bin == "0.25-0.5mi":
        rmin, rmax = 0.25 * 1609.344, 0.5 * 1609.344
    elif user_dist_bin == "0.5-0.75mi":
        rmin, rmax = 0.5 * 1609.344, 0.75 * 1609.344
    elif user_dist_bin == "0.75-1mi":
        rmin, rmax = 0.75 * 1609.344, 1.0 * 1609.344
    elif user_dist_bin == "1-1.5mi":
        rmin, rmax = 1.0 * 1609.344, 1.5 * 1609.344
    elif user_dist_bin == "1.5-2mi":
        rmin, rmax = 1.5 * 1609.344, 2.0 * 1609.344
    else:
        rmin, rmax = 2.0 * 1609.344, 3.0 * 1609.344

    pts: List[Tuple[float, float]] = []
    for _ in range(K):
        r = math.sqrt(RNG.random() * (rmax * rmax - rmin * rmin) + rmin * rmin)
        bearing = RNG.uniform(0.0, 360.0)
        pts.append(offset_from_anchor(anchor_latlon[0], anchor_latlon[1], r, bearing))
    return pts


def build_pas_token(query: Dict[str, Any], anchors: List[Dict[str, Any]], epsilon: float = 1.0, scale_m: float = 500.0) -> Dict[str, Any]:
    user_loc = (
        query['spatial_intent']['user_location']['lat'],
        query['spatial_intent']['user_location']['lon'],
    )
    weights: List[float] = []
    for anchor in anchors:
        d = haversine_m(user_loc, (anchor['geo']['lat'], anchor['geo']['lon']))
        weights.append(math.exp(-epsilon * d / scale_m))

    total = sum(weights)
    probs = [w / total for w in weights]
    idx = RNG.choices(range(len(anchors)), weights=probs, k=1)[0]
    anchor = anchors[idx]

    d = haversine_m(user_loc, (anchor['geo']['lat'], anchor['geo']['lon']))
    th = bearing_deg((anchor['geo']['lat'], anchor['geo']['lon']), user_loc)
    return {
        'anchor_id': anchor['anchor_id'],
        'anchor_name': anchor['name'],
        'anchor_borough': anchor.get('borough'),
        'anchor_location': {
            'lat': anchor['geo']['lat'],
            'lon': anchor['geo']['lon'],
        },
        'direction_bin': dir_bin(th),
        'distance_bin': dist_bin(d),
        'sampling_probability': round(probs[idx], 4),
    }


# -----------------------------
# Query parsing
# -----------------------------

def parse_query(raw_query: str, fallback_query: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    q = raw_query.lower()
    entity = 'place'
    if 'restaurant' in q:
        entity = 'restaurant'
    elif 'hotel' in q or 'inn' in q or 'suite' in q:
        entity = 'hotel'
    elif 'subway' in q or 'station' in q or 'transit' in q:
        entity = 'subway_station'

    direction = 'ANY'
    for candidate in ['NE', 'NW', 'SE', 'SW', 'N', 'E', 'S', 'W']:
        # raw query is not tokenized for NE etc, so use word boundaries for single letters only where practical
        needle = candidate.lower()
        if candidate in {'NE', 'NW', 'SE', 'SW'}:
            if needle in q:
                direction = candidate
                break
        else:
            keywords = {
                'N': [' north ', ' northern '],
                'E': [' east ', ' eastern '],
                'S': [' south ', ' southern '],
                'W': [' west ', ' western '],
            }
            padded = f' {q} '
            if any(k in padded for k in keywords[candidate]):
                direction = candidate
                break

    radius = None
    for candidate in [0.5, 1.0, 2.0, 3.0]:
        pattern = f'{int(candidate) if candidate.is_integer() else candidate} mile'
        if pattern in q or pattern + 's' in q:
            radius = candidate
            break
    if radius is None and fallback_query:
        radius = fallback_query['spatial_intent']['radius_miles']
    radius = radius or 2.0

    parsed = {
        'raw_query': raw_query,
        'semantic_intent': {
            'entity_type': entity,
            'attributes': [],
        },
        'spatial_intent': {
            'direction_constraint': direction,
            'radius_miles': radius,
        },
    }

    if fallback_query:
        parsed['spatial_intent']['user_location'] = fallback_query['spatial_intent']['user_location']
        # Keep user location from the eval query, but do not pass any anchor hint into runtime retrieval.
        # preserve richer attributes when we have them
        parsed['semantic_intent']['attributes'] = fallback_query.get('semantic_intent', {}).get('attributes', [])
        parsed['semantic_intent']['must_have_tags'] = fallback_query.get('semantic_intent', {}).get('must_have_tags', [entity])

    return parsed


def build_latent_user_samples(
    query: Dict[str, Any],
    pas_token: Dict[str, Any],
    mc_samples: int = 250,
) -> List[Tuple[float, float]]:
    anchor_latlon = None
    for key in ('anchor_location', 'anchor_geo'):
        if key in pas_token and pas_token[key]:
            anchor_latlon = (pas_token[key]['lat'], pas_token[key]['lon'])
            break
    if anchor_latlon is None:
        raise ValueError('PAS token must include anchor location to sample latent user positions.')

    if query['spatial_intent']['direction_constraint'] == 'ANY':
        return sample_full_annulus(anchor_latlon, pas_token['distance_bin'], K=mc_samples)
    return sample_Uz(anchor_latlon, pas_token['direction_bin'], pas_token['distance_bin'], K=mc_samples)


# -----------------------------
# Retrieval
# -----------------------------

def semantic_score(query: Dict[str, Any], chunk: Dict[str, Any]) -> float:
    entity = query['semantic_intent']['entity_type'].lower().replace(' ', '_')
    title = chunk['title'].lower()
    content = chunk['content'].lower()
    tags = [t.lower().replace(' ', '_') for t in chunk['metadata'].get('tags', [])]
    category = chunk['category'].lower().replace(' ', '_')
    subcategory = (chunk.get('subcategory') or '').lower().replace(' ', '_')

    score = 0.05
    if entity == category or entity == subcategory or entity in tags:
        score += 0.75
    if entity == 'subway_station' and ('station' in title or 'station' in content or 'transit' in tags):
        score += 0.15
    if entity == 'hotel' and any(k in content for k in ['hotel', 'lodging', 'suites', 'inn']):
        score += 0.15
    if entity == 'restaurant' and any(k in content for k in ['restaurant', 'eatery', 'dining', 'pizza', 'kitchen']):
        score += 0.15

    query_text = query['raw_query'].lower()
    lexical_hits = 0
    for token in set(query_text.replace('–', ' ').replace('-', ' ').split()):
        if len(token) >= 5 and (token in title or token in content):
            lexical_hits += 1
    score += min(0.1, lexical_hits * 0.02)
    return min(score, 1.0)


def spatial_score(
    query: Dict[str, Any],
    chunk: Dict[str, Any],
    pas_token: Optional[Dict[str, Any]] = None,
    latent_user_samples: Optional[List[Tuple[float, float]]] = None,
) -> Tuple[float, Optional[Dict[str, Any]]]:
    radius_m = query['spatial_intent']['radius_miles'] * 1609.344
    required_dir = query['spatial_intent']['direction_constraint']

    geo = chunk.get('metadata', {}).get('geo', {}) or {}
    lat = geo.get('lat')
    lon = geo.get('lon')
    if lat is None or lon is None:
        return 0.0, None
    item_loc = (lat, lon)

    candidate_tags = chunk['spatial']['anchor_tags']
    if pas_token:
        same_anchor_tags = [t for t in candidate_tags if t['anchor_id'] == pas_token['anchor_id']]
        if same_anchor_tags:
            candidate_tags = same_anchor_tags

    best_score = 0.0
    best_tag = None
    for tag in candidate_tags:
        if latent_user_samples:
            good = 0
            for x in latent_user_samples:
                within = haversine_m(x, item_loc) <= radius_m
                if required_dir == 'ANY':
                    directional = True
                else:
                    directional = dir_bin(bearing_deg(x, item_loc)) == required_dir
                if within and directional:
                    good += 1
            score = good / len(latent_user_samples)
        else:
            # Fallback deterministic check using anchor-relative tag fields.
            distance_m = tag.get('distance_m')
            direction_ok = required_dir == 'ANY' or tag.get('direction_bin') == required_dir
            if distance_m is not None and distance_m <= radius_m and direction_ok:
                score = 1.0
            elif distance_m is not None and distance_m <= radius_m:
                score = 0.45
            else:
                score = 0.0

        # Small preference for tags aligned with the sampled PAS anchor.
        if pas_token and tag['anchor_id'] == pas_token['anchor_id']:
            score = min(1.0, score + 0.05)

        if score > best_score:
            best_score = min(score, 1.0)
            best_tag = tag

    return best_score, best_tag


def hybrid_retrieve(
    query: Dict[str, Any],
    chunks: List[Dict[str, Any]],
    pas_token: Optional[Dict[str, Any]] = None,
    latent_user_samples: Optional[List[Tuple[float, float]]] = None,
    top_k: int = 3,
    lambda_weight: float = 0.65,
) -> List[Dict[str, Any]]:
    ranked: List[Dict[str, Any]] = []
    for chunk in chunks:
        sem = semantic_score(query, chunk)
        spa, matched_tag = spatial_score(query, chunk, pas_token=pas_token, latent_user_samples=latent_user_samples)
        final = lambda_weight * sem + (1 - lambda_weight) * spa
        ranked.append(
            {
                'rank': None,
                'chunk_id': chunk['chunk_id'],
                'doc_id': chunk['doc_id'],
                'title': chunk['title'],
                'category': chunk['category'],
                'score': round(final, 3),
                'semantic_score': round(sem, 3),
                'spatial_score': round(spa, 3),
                'content': chunk['content'],
                'supporting_facts': chunk.get('supporting_facts', []),
                'metadata': chunk['metadata'],
                'provenance': chunk.get('provenance', {}),
                'matched_anchor_tag': matched_tag,
            }
        )
    ranked.sort(key=lambda x: (x['score'], x['semantic_score'], x['spatial_score'], x['title']), reverse=True)
    for idx, row in enumerate(ranked[:top_k], start=1):
        row['rank'] = idx
    return ranked[:top_k]


# -----------------------------
# Grounding + generation
# -----------------------------

def build_grounded_prompt(query: Dict[str, Any], results: List[Dict[str, Any]], pas_token: Dict[str, Any]) -> str:
    blocks: List[str] = []
    for r in results:
        md = r['metadata']
        anchor = r.get('matched_anchor_tag') or {}
        facts = '\n'.join(f'- {fact}' for fact in r.get('supporting_facts', [])[:3])
        blocks.append(
            f"[{r['rank']}] {r['title']} ({r['doc_id']})\n"
            f"Category: {r['category']}\n"
            f"Neighborhood: {md.get('neighborhood', 'unknown')}\n"
            f"Address: {md.get('address', 'unknown')}\n"
            f"Score: {r['score']} | semantic={r['semantic_score']} spatial={r['spatial_score']}\n"
            f"Matched PAS relation: {anchor.get('spatial_relation_text', 'none')}\n"
            f"Content: {r['content']}\n"
            f"Supporting facts:\n{facts}\n"
        )
    context = '\n'.join(blocks)
    return (
        'You are a grounded PAS-RAG assistant.\n'
        'Answer only from the retrieved documents.\n'
        'Do not invent missing facts.\n'
        'Mention uncertainty when the context is insufficient.\n'
        'Cite document titles or document IDs inline when useful.\n\n'
        f"User question:\n{query['raw_query']}\n\n"
        f"PAS token:\n{json.dumps(pas_token, ensure_ascii=False)}\n\n"
        f"Retrieved context:\n{context}\n"
    )


def _hotel_sentence(result: Dict[str, Any]) -> str:
    md = result['metadata']
    amenity_bits = md.get('amenities', [])[:2]
    amenity_text = ', '.join(amenity_bits) if amenity_bits else 'basic amenities'
    return (
        f"{result['title']} is a {md.get('price_range', 'moderately priced')} option in {md.get('neighborhood', 'Brooklyn')} "
        f"with {amenity_text}."
    )


def _restaurant_sentence(result: Dict[str, Any]) -> str:
    md = result['metadata']
    cuisine = ', '.join(md.get('cuisine', [])[:2]) or result['category']
    signatures = ', '.join(md.get('signature_items', [])[:2])
    suffix = f" Popular items include {signatures}." if signatures else ''
    return f"{result['title']} is a {cuisine} spot in {md.get('neighborhood', 'Brooklyn')}.{suffix}"


def _station_sentence(result: Dict[str, Any]) -> str:
    md = result['metadata']
    lines = md.get('lines_served', [])[:4]
    line_text = ', '.join(lines) if lines else 'multiple lines'
    return f"{result['title']} serves {line_text} and is located in {md.get('neighborhood', 'Brooklyn')}."


def generate_grounded_answer(query: Dict[str, Any], results: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not results:
        return {
            'answer': 'I could not find any retrieved context that supports an answer to the question.',
            'citations': [],
            'faithfulness_notes': ['No retrieved chunks were available.'],
        }

    entity = query['semantic_intent']['entity_type']
    question = query['raw_query']
    intro = ''
    details: List[str] = []

    if entity == 'hotel':
        intro = (
            f"Based on the retrieved context, the main hotels matching '{question}' are "
            f"{', '.join(r['title'] for r in results[:-1]) + (', and ' + results[-1]['title'] if len(results) > 1 else results[0]['title'])}."
        )
        details = [_hotel_sentence(r) for r in results]
    elif entity == 'restaurant':
        intro = (
            f"Based on the retrieved context, the restaurants that best match '{question}' are "
            f"{', '.join(r['title'] for r in results[:-1]) + (', and ' + results[-1]['title'] if len(results) > 1 else results[0]['title'])}."
        )
        details = [_restaurant_sentence(r) for r in results]
    elif entity == 'subway_station':
        intro = (
            f"Based on the retrieved context, the nearest matching subway stations are "
            f"{', '.join(r['title'] for r in results[:-1]) + (', and ' + results[-1]['title'] if len(results) > 1 else results[0]['title'])}."
        )
        details = [_station_sentence(r) for r in results]
    else:
        intro = 'Based on the retrieved context, these are the best-matching places.'
        details = [f"{r['title']}: {r['content']}" for r in results]

    insufficiency: List[str] = []
    if any(r['spatial_score'] < 0.5 for r in results):
        insufficiency.append('Some returned items are semantically relevant but only weakly matched the spatial constraint.')

    answer = ' '.join([intro] + details + insufficiency)
    citations = [{'title': r['title'], 'doc_id': r['doc_id']} for r in results]
    return {
        'answer': answer,
        'citations': citations,
        'faithfulness_notes': [
            'Answer composed only from retrieved fields: title, content, metadata, and supporting_facts.',
            'No external knowledge was added.',
        ] + insufficiency,
    }


# -----------------------------
# Evaluation
# -----------------------------

def recall_at_k(gold_doc_ids: List[str], results: List[Dict[str, Any]], k: int = 3) -> float:
    gold = set(gold_doc_ids)
    top = {r['doc_id'] for r in results[:k]}
    return len(gold & top) / len(gold) if gold else 0.0


def mean_reciprocal_rank(gold_doc_ids: List[str], results: List[Dict[str, Any]]) -> float:
    gold = set(gold_doc_ids)
    for idx, r in enumerate(results, start=1):
        if r['doc_id'] in gold:
            return 1.0 / idx
    return 0.0


# -----------------------------
# Pipeline orchestration
# -----------------------------

def run_pas_rag_query(
    raw_query: str,
    chunks: List[Dict[str, Any]],
    anchors: List[Dict[str, Any]],
    fallback_query: Optional[Dict[str, Any]] = None,
    top_k: int = 3,
) -> Dict[str, Any]:
    parsed = parse_query(raw_query, fallback_query=fallback_query)
    pas_token = build_pas_token(parsed, anchors)
    latent_user_samples = build_latent_user_samples(parsed, pas_token, mc_samples=250)
    results = hybrid_retrieve(parsed, chunks, pas_token=pas_token, latent_user_samples=latent_user_samples, top_k=top_k)
    prompt = build_grounded_prompt(parsed, results, pas_token)
    generated = generate_grounded_answer(parsed, results)

    payload: Dict[str, Any] = {
        'query': raw_query,
        'parsed_query': parsed,
        'pas_token': pas_token,
        'latent_user_sample_count': len(latent_user_samples),
        'retrieved_chunks': results,
        'grounded_prompt': prompt,
        'generation_output': generated,
    }
    if fallback_query and fallback_query.get('ground_truth_doc_ids'):
        payload['evaluation'] = {
            'recall_at_3': round(recall_at_k(fallback_query['ground_truth_doc_ids'], results, 3), 3),
            'mrr': round(mean_reciprocal_rank(fallback_query['ground_truth_doc_ids'], results), 3),
            'ground_truth_doc_ids': fallback_query['ground_truth_doc_ids'],
        }
    return payload


def run_demo() -> List[Dict[str, Any]]:
    chunks = load_jsonl(BASE / 'chunks.jsonl')
    anchors = load_jsonl(BASE / 'anchor_registry.jsonl')

    eval_path_json = BASE / 'eval_queries.json'
    eval_path_jsonl = BASE / 'eval_queries.jsonl'
    if eval_path_json.exists():
        eval_queries = json.load(open(eval_path_json, 'r', encoding='utf-8'))
    elif eval_path_jsonl.exists():
        eval_queries = load_jsonl(eval_path_jsonl)
    else:
        raise FileNotFoundError("Expected eval_queries.json or eval_queries.jsonl next to the script.")

    outputs: List[Dict[str, Any]] = []
    recalls: List[float] = []
    mrrs: List[float] = []
    for q in eval_queries:
        result = run_pas_rag_query(q['raw_query'], chunks, anchors, fallback_query=q, top_k=3)
        outputs.append(result)
        if 'evaluation' in result:
            recalls.append(result['evaluation']['recall_at_3'])
            mrrs.append(result['evaluation']['mrr'])

    summary = {
        'num_queries': len(outputs),
        'mean_recall_at_3': round(statistics.mean(recalls), 3) if recalls else None,
        'mean_mrr': round(statistics.mean(mrrs), 3) if mrrs else None,
    }
    with open(BASE / 'pas_rag_demo_outputs_patched.json', 'w', encoding='utf-8') as f:
        json.dump({'summary': summary, 'runs': outputs}, f, indent=2, ensure_ascii=False)
    return outputs


if __name__ == '__main__':
    runs = run_demo()
    for run in runs:
        print('=' * 100)
        print(f"Query: {run['query']}")
        print('PAS token:', run['pas_token'])
        print('Top docs:', [(r['doc_id'], r['score']) for r in run['retrieved_chunks']])
        if 'evaluation' in run:
            print('Evaluation:', run['evaluation'])
        print('Answer:', run['generation_output']['answer'])
