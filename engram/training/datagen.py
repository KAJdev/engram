"""training data generation: llm synthetic data and small procedural demo dataset."""

from __future__ import annotations

import json
import random
import asyncio
from pathlib import Path
from dataclasses import dataclass, field, asdict

from engram.config import DataGenConfig


@dataclass
class Memory:
    text: str
    user_id: int
    memory_idx: int
    entities: list[str] = field(default_factory=list)
    themes: list[str] = field(default_factory=list)
    timestamp_days: int = 0  # day offset from user creation


@dataclass
class EdgeLabel:
    user_id: int
    memory_a_idx: int
    memory_b_idx: int
    edge_exists: bool
    edge_type: str  # complementary, causal, temporal_sequence, contradictory, elaborative, entity_overlap, none
    edge_weight: float
    implication: str = ""  # for complementary edges: what the combo implies


@dataclass
class RetrievalLabel:
    user_id: int
    query: str
    directly_relevant: list[int]  # memory indices
    jointly_relevant: list[list[int]]  # groups of indices only relevant together


@dataclass
class SynthesisLabel:
    user_id: int
    memory_a_idx: int
    memory_b_idx: int
    queries: list[str]  # queries where both are needed


# procedural demo dataset, no llm required

def generate_demo_dataset(output_dir: Path, seed: int = 42) -> dict:
    """generate a small but realistic dataset for pipeline validation."""
    random.seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    users = _generate_demo_users()
    all_memories: list[Memory] = []
    all_edges: list[EdgeLabel] = []
    all_retrieval: list[RetrievalLabel] = []
    all_synthesis: list[SynthesisLabel] = []

    for user_id, user_data in enumerate(users):
        memories = user_data["memories"]
        for i, m in enumerate(memories):
            all_memories.append(Memory(
                text=m["text"],
                user_id=user_id,
                memory_idx=i,
                entities=m.get("entities", []),
                themes=m.get("themes", []),
                timestamp_days=m.get("day", i),
            ))

        for edge in user_data["edges"]:
            all_edges.append(EdgeLabel(
                user_id=user_id,
                memory_a_idx=edge["a"],
                memory_b_idx=edge["b"],
                edge_exists=True,
                edge_type=edge["type"],
                edge_weight=edge.get("weight", 0.8),
                implication=edge.get("implication", ""),
            ))

        # negative edges: unlabeled random pairs
        labeled_pairs = {(e["a"], e["b"]) for e in user_data["edges"]}
        neg_count = len(user_data["edges"]) * 3
        for _ in range(neg_count):
            a = random.randint(0, len(memories) - 1)
            b = random.randint(0, len(memories) - 1)
            if a != b and (a, b) not in labeled_pairs and (b, a) not in labeled_pairs:
                all_edges.append(EdgeLabel(
                    user_id=user_id, memory_a_idx=a, memory_b_idx=b,
                    edge_exists=False, edge_type="none", edge_weight=0.0,
                ))

        for ret in user_data["retrieval"]:
            all_retrieval.append(RetrievalLabel(
                user_id=user_id,
                query=ret["query"],
                directly_relevant=ret["direct"],
                jointly_relevant=ret.get("joint", []),
            ))

        for synth in user_data.get("synthesis", []):
            all_synthesis.append(SynthesisLabel(
                user_id=user_id,
                memory_a_idx=synth["a"],
                memory_b_idx=synth["b"],
                queries=synth["queries"],
            ))

    _save_jsonl(output_dir / "memories.jsonl", [asdict(m) for m in all_memories])
    _save_jsonl(output_dir / "edges.jsonl", [asdict(e) for e in all_edges])
    _save_jsonl(output_dir / "retrieval.jsonl", [asdict(r) for r in all_retrieval])
    _save_jsonl(output_dir / "synthesis.jsonl", [asdict(s) for s in all_synthesis])

    stats = {
        "num_users": len(users),
        "num_memories": len(all_memories),
        "num_positive_edges": sum(1 for e in all_edges if e.edge_exists),
        "num_negative_edges": sum(1 for e in all_edges if not e.edge_exists),
        "num_retrieval_queries": len(all_retrieval),
        "num_synthesis_pairs": len(all_synthesis),
    }
    with open(output_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    return stats


def _generate_demo_users() -> list[dict]:
    """hand crafted synthetic users with known relationships."""
    users = []

    # user 0: medical scenario
    users.append({
        "memories": [
            {"text": "My doctor put me on warfarin for my blood clot last month", "entities": ["warfarin", "blood clot"], "themes": ["health", "medication"], "day": 0},
            {"text": "I've been taking St. John's Wort for my mood lately", "entities": ["st johns wort"], "themes": ["health", "medication", "mental health"], "day": 15},
            {"text": "I love going for morning runs in the park", "entities": ["park"], "themes": ["exercise", "routine"], "day": 3},
            {"text": "My sister's wedding is next month and I'm the best man", "entities": ["sister"], "themes": ["family", "events"], "day": 20},
            {"text": "I switched to a vegetarian diet three weeks ago", "entities": [], "themes": ["diet", "health"], "day": 10},
            {"text": "Been having trouble sleeping, might try melatonin", "entities": ["melatonin"], "themes": ["health", "sleep"], "day": 25},
            {"text": "My blood pressure has been running a bit high lately", "entities": ["blood pressure"], "themes": ["health"], "day": 18},
            {"text": "I have a dentist appointment next Tuesday", "entities": ["dentist"], "themes": ["health", "appointments"], "day": 28},
            {"text": "Started a new project at work managing the cloud migration", "entities": ["cloud migration"], "themes": ["work", "technology"], "day": 5},
            {"text": "I need to pick up my prescription refill from CVS", "entities": ["CVS", "prescription"], "themes": ["health", "medication", "errands"], "day": 22},
        ],
        "edges": [
            {"a": 0, "b": 1, "type": "complementary", "weight": 0.95, "implication": "Dangerous drug interaction: St. John's Wort reduces warfarin effectiveness, increasing clotting risk"},
            {"a": 0, "b": 7, "type": "complementary", "weight": 0.8, "implication": "Dentist needs to know about warfarin due to bleeding risk during procedures"},
            {"a": 0, "b": 9, "type": "elaborative", "weight": 0.7},
            {"a": 1, "b": 5, "type": "entity_overlap", "weight": 0.5},
            {"a": 0, "b": 6, "type": "elaborative", "weight": 0.6},
            {"a": 4, "b": 0, "type": "complementary", "weight": 0.7, "implication": "Vegetarian diet changes vitamin K intake which affects warfarin dosing"},
        ],
        "retrieval": [
            {"query": "I have a terrible headache, should I just take ibuprofen?", "direct": [0], "joint": [[0, 1]]},
            {"query": "What medications am I currently taking?", "direct": [0, 1, 5], "joint": []},
            {"query": "Do I have any upcoming appointments?", "direct": [3, 7], "joint": []},
            {"query": "Should I be worried about anything health-related?", "direct": [0, 6], "joint": [[0, 1], [0, 4]]},
        ],
        "synthesis": [
            {"a": 0, "b": 1, "queries": ["should I take ibuprofen for my headache", "are there any risks with my current medications", "what should I tell my doctor at my next visit"]},
            {"a": 0, "b": 7, "queries": ["what should I tell my dentist", "is it safe to have dental work done"]},
            {"a": 0, "b": 4, "queries": ["should I change my diet while on medication", "does my new diet affect my treatment"]},
        ],
    })

    # user 1: career and skills
    users.append({
        "memories": [
            {"text": "I manage a team of 5 software engineers", "entities": ["team"], "themes": ["work", "management"], "day": 0},
            {"text": "We're planning to migrate our infrastructure to GCP", "entities": ["GCP"], "themes": ["work", "technology", "cloud"], "day": 5},
            {"text": "Two of my engineers only have AWS experience", "entities": ["AWS"], "themes": ["work", "skills", "cloud"], "day": 8},
            {"text": "I've been learning Kubernetes in my spare time", "entities": ["Kubernetes"], "themes": ["technology", "learning"], "day": 12},
            {"text": "The migration deadline is end of Q2", "entities": ["Q2"], "themes": ["work", "deadlines"], "day": 15},
            {"text": "My wife and I are expecting our first child in June", "entities": ["wife", "child"], "themes": ["family", "life events"], "day": 10},
            {"text": "I've been feeling burned out from the long hours", "entities": [], "themes": ["work", "mental health", "burnout"], "day": 20},
            {"text": "Our team's sprint velocity has dropped 30% this quarter", "entities": ["sprint velocity"], "themes": ["work", "productivity"], "day": 22},
            {"text": "I got a LinkedIn message from a recruiter at a startup", "entities": ["LinkedIn", "recruiter"], "themes": ["career", "opportunity"], "day": 25},
            {"text": "My performance review is coming up next month", "entities": ["performance review"], "themes": ["work", "career"], "day": 27},
        ],
        "edges": [
            {"a": 1, "b": 2, "type": "complementary", "weight": 0.9, "implication": "Skills gap risk: team needs GCP skills but key members only know AWS"},
            {"a": 0, "b": 7, "type": "causal", "weight": 0.7},
            {"a": 1, "b": 4, "type": "temporal_sequence", "weight": 0.8},
            {"a": 4, "b": 5, "type": "complementary", "weight": 0.85, "implication": "Migration deadline and baby due date both in Q2 - major capacity conflict"},
            {"a": 6, "b": 7, "type": "causal", "weight": 0.75},
            {"a": 6, "b": 8, "type": "causal", "weight": 0.6},
            {"a": 3, "b": 1, "type": "elaborative", "weight": 0.65},
            {"a": 9, "b": 7, "type": "complementary", "weight": 0.7, "implication": "Performance review coming while team velocity is down"},
        ],
        "retrieval": [
            {"query": "What risks should I think about for the migration?", "direct": [1, 4], "joint": [[1, 2], [4, 5]]},
            {"query": "Should I consider the recruiter's offer?", "direct": [8], "joint": [[6, 7, 8]]},
            {"query": "How should I prepare for my performance review?", "direct": [9], "joint": [[7, 9], [0, 7]]},
        ],
        "synthesis": [
            {"a": 1, "b": 2, "queries": ["what training does my team need", "what are the biggest risks for the GCP migration"]},
            {"a": 4, "b": 5, "queries": ["how should I plan my Q2 schedule", "will I have enough bandwidth for the migration"]},
            {"a": 9, "b": 7, "queries": ["what should I highlight in my review", "am I at risk in my performance review"]},
        ],
    })

    # user 2: travel and dietary
    users.append({
        "memories": [
            {"text": "I've been vegetarian for five years now", "entities": [], "themes": ["diet", "lifestyle"], "day": 0},
            {"text": "Planning a three-week trip to rural Japan next spring", "entities": ["Japan"], "themes": ["travel", "planning"], "day": 5},
            {"text": "I'm severely allergic to shellfish", "entities": ["shellfish"], "themes": ["health", "diet", "allergy"], "day": 2},
            {"text": "I speak basic conversational Japanese from college", "entities": ["Japanese"], "themes": ["language", "skills"], "day": 3},
            {"text": "My budget for the trip is about $4000", "entities": [], "themes": ["travel", "finance"], "day": 7},
            {"text": "I practice yoga every morning without fail", "entities": ["yoga"], "themes": ["exercise", "routine"], "day": 1},
            {"text": "I recently got certified as a scuba diver", "entities": ["scuba"], "themes": ["hobbies", "sports"], "day": 15},
            {"text": "My partner doesn't eat gluten", "entities": ["partner"], "themes": ["diet", "relationships"], "day": 8},
            {"text": "I've been reading Haruki Murakami novels obsessively", "entities": ["Murakami"], "themes": ["hobbies", "literature", "Japan"], "day": 10},
            {"text": "I need to renew my passport before the trip", "entities": ["passport"], "themes": ["travel", "errands"], "day": 12},
        ],
        "edges": [
            {"a": 0, "b": 1, "type": "complementary", "weight": 0.9, "implication": "Being vegetarian in rural Japan is very challenging - limited options, dashi in everything"},
            {"a": 2, "b": 1, "type": "complementary", "weight": 0.85, "implication": "Shellfish allergy in Japan is dangerous - hidden shellfish in many dishes, need to communicate clearly"},
            {"a": 3, "b": 1, "type": "elaborative", "weight": 0.7},
            {"a": 3, "b": 2, "type": "complementary", "weight": 0.75, "implication": "Japanese language skills needed to communicate allergy - could be life-saving"},
            {"a": 7, "b": 1, "type": "complementary", "weight": 0.8, "implication": "Partner's gluten restriction adds another dietary constraint for Japan trip"},
            {"a": 8, "b": 1, "type": "entity_overlap", "weight": 0.5},
            {"a": 6, "b": 1, "type": "complementary", "weight": 0.6, "implication": "Could plan scuba diving in Okinawa during Japan trip"},
            {"a": 9, "b": 1, "type": "temporal_sequence", "weight": 0.7},
        ],
        "retrieval": [
            {"query": "Help me plan meals for my trip", "direct": [1], "joint": [[0, 1], [2, 1], [7, 1]]},
            {"query": "What should I pack for Japan?", "direct": [1], "joint": [[5, 1]]},
            {"query": "Are there any safety concerns for my trip?", "direct": [1, 9], "joint": [[2, 1, 3]]},
        ],
        "synthesis": [
            {"a": 0, "b": 1, "queries": ["where can I eat in rural Japan", "will I be able to find food I can eat"]},
            {"a": 2, "b": 1, "queries": ["what allergy card should I carry in Japan", "how do I explain my allergy in Japanese"]},
            {"a": 2, "b": 3, "queries": ["how do I say I'm allergic to shellfish in Japanese"]},
        ],
    })

    # user 3: emotional patterns
    users.append({
        "memories": [
            {"text": "Had a panic attack before my presentation at the all-hands meeting", "entities": ["presentation", "all-hands"], "themes": ["work", "mental health", "anxiety"], "day": 0},
            {"text": "Skipped the team offsite last weekend, said I was sick", "entities": ["team offsite"], "themes": ["work", "avoidance", "social"], "day": 10},
            {"text": "My manager mentioned I seem less engaged lately", "entities": ["manager"], "themes": ["work", "feedback"], "day": 15},
            {"text": "I've been invited to give a talk at the regional conference", "entities": ["conference"], "themes": ["work", "career", "public speaking"], "day": 20},
            {"text": "Started seeing a therapist for general anxiety", "entities": ["therapist"], "themes": ["mental health", "treatment"], "day": 12},
            {"text": "I really enjoy the coding parts of my job", "entities": [], "themes": ["work", "satisfaction"], "day": 5},
            {"text": "My gym routine has been really inconsistent this month", "entities": ["gym"], "themes": ["exercise", "routine", "wellbeing"], "day": 18},
            {"text": "I keep declining after-work social invitations", "entities": [], "themes": ["social", "avoidance"], "day": 22},
        ],
        "edges": [
            {"a": 0, "b": 3, "type": "complementary", "weight": 0.9, "implication": "Pattern: anxiety before presentations + now invited to conference talk = likely anxiety trigger"},
            {"a": 0, "b": 1, "type": "causal", "weight": 0.8},
            {"a": 1, "b": 2, "type": "causal", "weight": 0.7},
            {"a": 1, "b": 7, "type": "elaborative", "weight": 0.75},
            {"a": 0, "b": 4, "type": "causal", "weight": 0.65},
            {"a": 2, "b": 7, "type": "elaborative", "weight": 0.6},
            {"a": 6, "b": 0, "type": "complementary", "weight": 0.5, "implication": "Physical routine disruption often correlates with anxiety episodes"},
        ],
        "retrieval": [
            {"query": "Should I accept the conference speaking invitation?", "direct": [3], "joint": [[0, 3], [0, 4, 3]]},
            {"query": "What should I talk about with my therapist?", "direct": [4], "joint": [[0, 1, 2, 7]]},
            {"query": "Why do I feel so disconnected from work?", "direct": [2, 5], "joint": [[1, 2, 7]]},
        ],
        "synthesis": [
            {"a": 0, "b": 3, "queries": ["will I be able to handle the conference talk", "should I prepare differently for this presentation"]},
        ],
    })

    # user 4: recommendations and preferences
    users.append({
        "memories": [
            {"text": "I absolutely loved the movie Arrival - the way it handles time and language was incredible", "entities": ["Arrival"], "themes": ["entertainment", "preferences", "sci-fi"], "day": 0},
            {"text": "Been reading a lot of Jorge Luis Borges short stories", "entities": ["Borges"], "themes": ["literature", "preferences"], "day": 5},
            {"text": "I prefer slow, thoughtful content over action-packed stuff", "entities": [], "themes": ["preferences", "entertainment"], "day": 8},
            {"text": "Just finished Annihilation by Jeff VanderMeer and loved it", "entities": ["Annihilation", "VanderMeer"], "themes": ["literature", "preferences", "sci-fi"], "day": 12},
            {"text": "I find most Marvel movies pretty boring honestly", "entities": ["Marvel"], "themes": ["preferences", "entertainment"], "day": 3},
            {"text": "Stalker by Tarkovsky is my favorite film of all time", "entities": ["Stalker", "Tarkovsky"], "themes": ["entertainment", "preferences", "film"], "day": 1},
            {"text": "I've been getting into ambient electronic music lately", "entities": [], "themes": ["music", "preferences"], "day": 15},
            {"text": "Tried watching the latest Netflix action thriller, turned it off after 20 minutes", "entities": ["Netflix"], "themes": ["entertainment", "preferences"], "day": 18},
        ],
        "edges": [
            {"a": 0, "b": 1, "type": "complementary", "weight": 0.8, "implication": "Both involve themes of time, infinity, and language - clear aesthetic pattern"},
            {"a": 0, "b": 5, "type": "entity_overlap", "weight": 0.6},
            {"a": 2, "b": 4, "type": "elaborative", "weight": 0.7},
            {"a": 2, "b": 7, "type": "elaborative", "weight": 0.7},
            {"a": 0, "b": 3, "type": "entity_overlap", "weight": 0.65},
            {"a": 3, "b": 5, "type": "complementary", "weight": 0.7, "implication": "Both atmospheric, mysterious sci-fi - VanderMeer and Tarkovsky share aesthetic sensibility"},
            {"a": 6, "b": 2, "type": "elaborative", "weight": 0.5},
        ],
        "retrieval": [
            {"query": "What should I watch tonight?", "direct": [2], "joint": [[0, 2, 5], [4, 7]]},
            {"query": "Can you recommend a book for me?", "direct": [], "joint": [[1, 3, 2]]},
            {"query": "What kind of content do I like?", "direct": [2, 4], "joint": [[0, 1, 3, 5, 6]]},
        ],
        "synthesis": [
            {"a": 0, "b": 1, "queries": ["what movie should I watch", "what book should I read next"]},
            {"a": 3, "b": 5, "queries": ["recommend something atmospheric and mysterious"]},
        ],
    })

    # user 5: social connections
    users.append({
        "memories": [
            {"text": "My sister is an immigration lawyer in Boston", "entities": ["sister", "immigration lawyer", "Boston"], "themes": ["family", "careers"], "day": 0},
            {"text": "My coworker Raj mentioned his H1B visa renewal is causing him stress", "entities": ["Raj", "H1B visa"], "themes": ["work", "immigration", "relationships"], "day": 10},
            {"text": "I play basketball with Dave every Thursday", "entities": ["Dave", "basketball"], "themes": ["social", "exercise", "routine"], "day": 2},
            {"text": "My neighbor just adopted a rescue dog from Guatemala", "entities": ["neighbor", "dog", "Guatemala"], "themes": ["community", "pets"], "day": 15},
            {"text": "I'm organizing the company holiday party this year", "entities": ["holiday party"], "themes": ["work", "events", "social"], "day": 20},
            {"text": "Dave's wife is a veterinarian", "entities": ["Dave", "veterinarian"], "themes": ["social", "careers"], "day": 5},
            {"text": "Raj is one of our best engineers and I'd hate to lose him", "entities": ["Raj"], "themes": ["work", "team"], "day": 12},
        ],
        "edges": [
            {"a": 0, "b": 1, "type": "complementary", "weight": 0.9, "implication": "Sister could potentially help or advise Raj with his immigration situation"},
            {"a": 1, "b": 6, "type": "elaborative", "weight": 0.8},
            {"a": 2, "b": 5, "type": "entity_overlap", "weight": 0.6},
            {"a": 3, "b": 5, "type": "complementary", "weight": 0.7, "implication": "Neighbor's new rescue dog + Dave's wife is a vet - could connect them"},
        ],
        "retrieval": [
            {"query": "Is there anything I can do to help Raj?", "direct": [1, 6], "joint": [[0, 1]]},
            {"query": "My neighbor's dog seems sick, who should they call?", "direct": [3], "joint": [[3, 5]]},
        ],
        "synthesis": [
            {"a": 0, "b": 1, "queries": ["can anyone I know help with visa issues", "how can I support Raj"]},
            {"a": 3, "b": 5, "queries": ["does anyone I know work with animals"]},
        ],
    })

    return users


def _save_jsonl(path: Path, data: list[dict]) -> None:
    with open(path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")


# llm based data generation for production scale training

SYNTHETIC_USER_PROMPT = """Generate {n_memories} memories for a fictional person over {months} months.

Include their job, health conditions, medications, relationships, hobbies, travel plans, daily routines, preferences, and evolving situations.

Make some memories obviously related, some subtly related (e.g., a food allergy mentioned once and a restaurant trip planned months later), and some unrelated.

Include at least {n_complementary} cases where two memories that seem unrelated have important combined implications (like drug interactions, schedule conflicts, preference contradictions, or skill-job relevance).

Return as JSON array where each memory has:
- "text": the memory text (first person, natural language)
- "entities": list of key entities mentioned
- "themes": list of life domains (health, work, relationships, finance, etc.)
- "day": integer day offset from first memory

Return ONLY the JSON array, no other text."""


EDGE_LABEL_PROMPT = """Here are two memories from the same person:
Memory A: {memory_a}
Memory B: {memory_b}

Are these memories related? Respond with JSON:
{{
  "edge_exists": true/false,
  "edge_type": "complementary" | "causal" | "temporal_sequence" | "contradictory" | "elaborative" | "entity_overlap" | "none",
  "edge_weight": 0.0-1.0,
  "implication": "if complementary, what does their combination imply that neither states alone?",
  "confidence": 0.0-1.0
}}

Edge types:
- complementary: combination implies something neither states alone (THIS IS THE MOST IMPORTANT TYPE)
- causal: one led to or explains the other
- temporal_sequence: events in a sequence
- contradictory: they conflict
- elaborative: one adds detail to the other
- entity_overlap: share entities but not otherwise deeply related
- none: unrelated

Return ONLY the JSON, no other text."""


RETRIEVAL_LABEL_PROMPT = """Here are memories for a person:
{memories}

Generate {n_queries} questions this person might ask, ranging from simple factual recall to complex questions where the answer depends on combining multiple memories.

For each question, return JSON with:
- "query": the question
- "direct": list of memory indices that are directly relevant
- "joint": list of groups of memory indices that are only relevant IN COMBINATION (e.g., [[0, 3], [1, 5]] means memories 0+3 are jointly relevant, and memories 1+5 are jointly relevant)

Return as JSON array. Return ONLY the JSON array, no other text."""


SYNTHESIS_LABEL_PROMPT = """Memory A: {memory_a}
Memory B: {memory_b}

These have a complementary relationship: {implication}

Write 3-5 queries where BOTH memories would be needed to give a good answer, but where NEITHER memory alone would be retrieved by standard keyword/semantic search.

Return as JSON array of strings. Return ONLY the JSON array, no other text."""


async def generate_llm_dataset(config: DataGenConfig) -> dict:
    """generate full training dataset using llm calls.
    providers: anthropic, openai, or vllm for open source models."""
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if config.llm_provider == "anthropic":
        from anthropic import AsyncAnthropic
        client = AsyncAnthropic()

        async def call_llm(prompt: str) -> str:
            resp = await client.messages.create(
                model=config.llm_model,
                max_tokens=8192,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.content[0].text

    elif config.llm_provider == "vllm":
        # vllm serves an openai compatible api, use openai client with custom base_url
        from openai import AsyncOpenAI
        client = AsyncOpenAI(
            base_url=config.vllm_url,
            api_key="not-needed",  # vllm doesnt need a key
        )

        async def call_llm(prompt: str) -> str:
            resp = await client.chat.completions.create(
                model=config.llm_model,
                max_tokens=4096,
                temperature=0.7,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.choices[0].message.content

    else:
        from openai import AsyncOpenAI
        client = AsyncOpenAI()

        async def call_llm(prompt: str) -> str:
            resp = await client.chat.completions.create(
                model=config.llm_model,
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.choices[0].message.content

    all_memories = []
    all_edges = []
    all_retrieval = []
    all_synthesis = []

    sem = asyncio.Semaphore(config.max_concurrent)

    async def gen_user(user_id: int) -> dict | None:
        async with sem:
            try:
                prompt = SYNTHETIC_USER_PROMPT.format(
                    n_memories=config.memories_per_user,
                    months=8,
                    n_complementary=10,
                )
                raw = await call_llm(prompt)
                memories = json.loads(raw)
                return {"user_id": user_id, "memories": memories}
            except Exception as e:
                print(f"Failed user {user_id}: {e}")
                return None

    # generate users
    tasks = [gen_user(i) for i in range(config.num_synthetic_users)]
    users = [u for u in await asyncio.gather(*tasks) if u is not None]

    for user in users:
        uid = user["user_id"]
        memories = user["memories"]

        for i, m in enumerate(memories):
            all_memories.append(Memory(
                text=m["text"], user_id=uid, memory_idx=i,
                entities=m.get("entities", []),
                themes=m.get("themes", []),
                timestamp_days=m.get("day", i),
            ))

        # edge labels for sampled pairs
        pairs = _sample_pairs(len(memories), config.pairs_per_user)
        edge_tasks = []
        for a, b in pairs:
            async def gen_edge(a=a, b=b):
                async with sem:
                    try:
                        prompt = EDGE_LABEL_PROMPT.format(
                            memory_a=memories[a]["text"],
                            memory_b=memories[b]["text"],
                        )
                        raw = await call_llm(prompt)
                        label = json.loads(raw)
                        return EdgeLabel(
                            user_id=uid, memory_a_idx=a, memory_b_idx=b,
                            edge_exists=label["edge_exists"],
                            edge_type=label["edge_type"],
                            edge_weight=label.get("edge_weight", 0.5),
                            implication=label.get("implication", ""),
                        )
                    except Exception as e:
                        print(f"Failed edge ({a},{b}) user {uid}: {e}")
                        return None
            edge_tasks.append(gen_edge())

        edges = [e for e in await asyncio.gather(*edge_tasks) if e is not None]
        all_edges.extend(edges)

        # retrieval labels
        async def gen_retrieval():
            async with sem:
                try:
                    mem_text = "\n".join(
                        f"[{i}] {m['text']}" for i, m in enumerate(memories)
                    )
                    prompt = RETRIEVAL_LABEL_PROMPT.format(
                        memories=mem_text, n_queries=config.queries_per_user,
                    )
                    raw = await call_llm(prompt)
                    labels = json.loads(raw)
                    return [
                        RetrievalLabel(
                            user_id=uid, query=l["query"],
                            directly_relevant=l["direct"],
                            jointly_relevant=l.get("joint", []),
                        )
                        for l in labels
                    ]
                except Exception as e:
                    print(f"Failed retrieval user {uid}: {e}")
                    return []

        retrieval = await gen_retrieval()
        all_retrieval.extend(retrieval)

        # synthesis labels for complementary edges
        comp_edges = [e for e in edges if e.edge_type == "complementary" and e.edge_exists]
        synth_tasks = []
        for edge in comp_edges:
            async def gen_synth(edge=edge):
                async with sem:
                    try:
                        prompt = SYNTHESIS_LABEL_PROMPT.format(
                            memory_a=memories[edge.memory_a_idx]["text"],
                            memory_b=memories[edge.memory_b_idx]["text"],
                            implication=edge.implication,
                        )
                        raw = await call_llm(prompt)
                        queries = json.loads(raw)
                        return SynthesisLabel(
                            user_id=uid,
                            memory_a_idx=edge.memory_a_idx,
                            memory_b_idx=edge.memory_b_idx,
                            queries=queries,
                        )
                    except Exception as e:
                        print(f"Failed synthesis user {uid}: {e}")
                        return None
            synth_tasks.append(gen_synth())

        synths = [s for s in await asyncio.gather(*synth_tasks) if s is not None]
        all_synthesis.extend(synths)

    _save_jsonl(output_dir / "memories.jsonl", [asdict(m) for m in all_memories])
    _save_jsonl(output_dir / "edges.jsonl", [asdict(e) for e in all_edges])
    _save_jsonl(output_dir / "retrieval.jsonl", [asdict(r) for r in all_retrieval])
    _save_jsonl(output_dir / "synthesis.jsonl", [asdict(s) for s in all_synthesis])

    stats = {
        "num_users": len(users),
        "num_memories": len(all_memories),
        "num_positive_edges": sum(1 for e in all_edges if e.edge_exists),
        "num_negative_edges": sum(1 for e in all_edges if not e.edge_exists),
        "num_retrieval_queries": len(all_retrieval),
        "num_synthesis_pairs": len(all_synthesis),
    }
    with open(output_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    return stats


def _sample_pairs(n: int, num_pairs: int) -> list[tuple[int, int]]:
    """sample random pairs of indices."""
    import random
    pairs = set()
    max_pairs = n * (n - 1) // 2
    num_pairs = min(num_pairs, max_pairs)
    while len(pairs) < num_pairs:
        a = random.randint(0, n - 1)
        b = random.randint(0, n - 1)
        if a != b:
            pairs.add((min(a, b), max(a, b)))
    return list(pairs)
