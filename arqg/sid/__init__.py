"""SID — Synthetic Index Dataset for agentic retrieval (plan v0.8, first version).

A task factory producing

    (question, versioned index, minimal-sufficient gold set,
     coverage/complexity labels, optionally injected distractors)

Stage map (plan §1 → module):

    S0  compat.py       index/unit compatibility, fields, versioning
    S1  subgraphs.py    entity ↔ chunk bipartite mining
    S2  taxonomy.py     A1 mechanic cell + difficulty sampling
    S3  facts.py        atomic facts with verbatim spans
        compose.py      1-of-N question composition
    S4  gates.py        G_SOLVE / G_BROAD / G_REACH
    S5  gates.py        G_MIN (leave-one-fact-out) / G_REP (fact groups)
    S6  density.py      neighbourhood density, τ_sim
        distractors.py  transplant → perturb → generate cascade
        inject.py       v0 → vN additive injection
    S7  isolation.py    cross-task isolation on the post-injection index
    S8  export.py       final tasks, splits, datamix stats

Teacher trajectories (plan §9.1) need the RL harness and are out of scope here;
`export.py` produces the task pool they would be collected on.
"""
