from typing import Dict, Any, List
import json
import re
from sambanova import SambaNova


with open("sambanova_api.txt", "r") as f:
    api_key = f.read().strip()

client = SambaNova(
    base_url="https://api.sambanova.ai/v1",
    api_key=api_key
)


def generate_grounded_answer(query: Dict[str, Any], results: List[Dict[str, Any]]) -> Dict[str, Any]:

    if not results:
        return {
            'answer': 'I could not find any retrieved context that supports an answer to the question.',
            'citations': [],
            'faithfulness_notes': ['No retrieved chunks were available.'],
        }

    # Build compact, grounded context from retrieved results only.
    context_blocks = []
    for i, r in enumerate(results, start=1):
        md = r.get('metadata', {}) or {}
        geo = md.get('geo', {}) or {}
        supporting = r.get('supporting_facts', []) or []
        context_blocks.append(
            f"""[DOC {i}]
doc_id: {r.get('doc_id', '')}
title: {r.get('title', '')}
category: {r.get('category', '')}
score: {r.get('score', 0.0)}
semantic_score: {r.get('semantic_score', 0.0)}
spatial_score: {r.get('spatial_score', 0.0)}
address: {md.get('address', '')}
neighborhood: {md.get('neighborhood', '')}
borough: {md.get('borough', '')}
tags: {', '.join(md.get('tags', []))}
geo: ({geo.get('lat', '')}, {geo.get('lon', '')})
content: {r.get('content', '')}
supporting_facts:
- """ + "\n- ".join(str(x) for x in supporting[:3])
        )

    context_text = "\n\n".join(context_blocks)

    entity = query.get('semantic_intent', {}).get('entity_type', 'place')
    must_have_tags = query.get('semantic_intent', {}).get('must_have_tags', []) or []
    spatial_intent = query.get('spatial_intent', {}) or {}
    radius_miles = spatial_intent.get('radius_miles', None)
    direction_constraint = spatial_intent.get('direction_constraint', 'ANY')

    system_prompt = (
        "You are a careful RAG answering assistant.\n"
        "Answer the user's query using ONLY the retrieved context.\n"
        "Do not use outside knowledge.\n"
        "If the evidence is weak or partially mismatched, say so briefly.\n"
        "Prefer concise answers that directly name the best matching places.\n"
        "Only cite documents that actually support your answer.\n"
        "Return STRICT JSON with exactly these keys:\n"
        "{\n"
        '  "answer": string,\n'
        '  "citations": [{"title": string, "doc_id": string}],\n'
        '  "faithfulness_notes": [string]\n'
        "}\n"
        "Do not include markdown fences."
    )

    user_prompt = f"""
User query: {query.get('raw_query', '')}

Semantic target:
- entity_type: {entity}
- must_have_tags: {must_have_tags}

Spatial intent:
- direction_constraint: {direction_constraint}
- radius_miles: {radius_miles}

Retrieved context:
{context_text}

Instructions:
1. Write a grounded answer that directly answers the user query.
2. Mention the strongest matching places first.
3. Exclude obviously unsupported claims.
4. If some retrieved items are only weak spatial matches, you may mention that uncertainty briefly.
5. Cite only the docs you actually used in the answer.
6. Output STRICT JSON only.
""".strip()

    # Safe deterministic fallback in case the API call or JSON parsing fails.
    def _fallback():
        titles = [r.get('title', '') for r in results]
        if len(titles) == 1:
            joined_titles = titles[0]
        elif len(titles) == 2:
            joined_titles = f"{titles[0]} and {titles[1]}"
        else:
            joined_titles = ", ".join(titles[:-1]) + f", and {titles[-1]}"

        answer = f"Based on the retrieved context, the best matching {entity}s are {joined_titles}."
        weak = [r for r in results if r.get('spatial_score', 0.0) < 0.5]
        notes = [
            'Fallback answer used because model output was unavailable or unparsable.',
            'Answer composed only from retrieved results.',
        ]
        if weak:
            notes.append('Some returned items are semantically relevant but only weakly matched the spatial constraint.')

        return {
            'answer': answer,
            'citations': [
                {'title': r.get('title', ''), 'doc_id': r.get('doc_id', '')}
                for r in results[: min(2, len(results))]
                if r.get('doc_id')
            ],
            'faithfulness_notes': notes,
        }

    try:
        completion = client.chat.completions.create(
            model="gpt-oss-120b",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=500,
        )

        content = completion.choices[0].message.content or ""
        content = content.strip()

        # Remove accidental code fences if present.
        if content.startswith("```"):
            content = re.sub(r"^```(?:json)?\s*", "", content)
            content = re.sub(r"\s*```$", "", content)

        parsed = json.loads(content)

        answer = parsed.get("answer", "")
        citations = parsed.get("citations", [])
        faithfulness_notes = parsed.get("faithfulness_notes", [])

        # Basic schema cleanup so the rest of your code keeps working.
        if not isinstance(answer, str):
            answer = str(answer)

        if not isinstance(citations, list):
            citations = []
        clean_citations = []
        seen_doc_ids = set()
        valid_doc_ids = {r.get('doc_id') for r in results if r.get('doc_id')}

        for c in citations:
            if not isinstance(c, dict):
                continue
            doc_id = c.get("doc_id")
            title = c.get("title", "")
            if not doc_id or doc_id not in valid_doc_ids or doc_id in seen_doc_ids:
                continue
            if not title:
                # Fill title from retrieved results if omitted
                title = next((r.get('title', '') for r in results if r.get('doc_id') == doc_id), '')
            clean_citations.append({"title": title, "doc_id": doc_id})
            seen_doc_ids.add(doc_id)

        if not isinstance(faithfulness_notes, list):
            faithfulness_notes = [str(faithfulness_notes)]

        if not answer.strip():
            return _fallback()

        if not clean_citations:
            # If the model failed to cite, keep only the strongest retrieved doc as a conservative fallback citation.
            best = results[0]
            clean_citations = [{
                "title": best.get("title", ""),
                "doc_id": best.get("doc_id", "")
            }] if best.get("doc_id") else []

        return {
            "answer": answer.strip(),
            "citations": clean_citations,
            "faithfulness_notes": faithfulness_notes + [
                "Answer generated from retrieved context only.",
                "Citations were restricted to retrieved doc_ids."
            ],
        }

    except Exception as e:
        fallback = _fallback()
        fallback["faithfulness_notes"].append(f"Generation fallback triggered: {type(e).__name__}")
        return fallback