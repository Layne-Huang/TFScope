# CLAUDE.md

Guidance for Claude Code in this repository.

---
You need to use mamba instead of conda.

You need to create tfscope environment and use this to run the code.

You need to end with "mu~" in every conversation. 

remmeber the cache dir is /n/holylabs/lpinello_lab/Lab/leihuang/.cache

All big files should be saved under /n/holylabs/lpinello_lab/Lab/leihuang/TFScope

# PART I: MULTI-AGENT RESEARCH FRAMEWORK

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  PHASE 1: LITERATURE & BRAINSTORMING                                        │
│  ├── Collect relevant papers                                                │
│  ├── Create specialized agents                                              │
│  └── Conduct review sessions (1:1 or lab meeting style)                     │
├─────────────────────────────────────────────────────────────────────────────┤
│  PHASE 2: IMPLEMENTATION                                                    │
│  ├── Write analysis code                                                    │
│  ├── Create visualizations                                                  │
│  └── Iterate with agent review loops                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│  PHASE 3: FIGURES & RESULTS                                                 │
│  ├── Generate publication-quality figures                                   │
│  └── Compile results and statistics                                         │
├─────────────────────────────────────────────────────────────────────────────┤
│  PHASE 4: PAPER WRITING                                                     │
│  ├── Draft manuscript sections                                              │
│  └── Iterate based on agent feedback                                        │
└─────────────────────────────────────────────────────────────────────────────┘
         ↑                                                           │
         └───────────── Iterate as needed ───────────────────────────┘
```

**Phase 1:** Collect papers in `papers/`, create agents via `skills/agent-generator/`, run brainstorming sessions.

**Phase 2:** Write code in `src/`, iterate with agent review. Exploratory work in notebooks, then refactor to scripts.

**Phase 3:** Output figures to `figures/`. Use colorblind-friendly palettes, vector formats (PDF/SVG).

**Phase 4:** Draft manuscript sections, review with agents, polish for submission.

**Collaboration patterns:** One-on-one deep dives, lab meeting presentations, review loops with critique and iteration.

## Skills

Reusable capabilities live in `skills/`. Each skill has a `SKILL.md` (instructions) and `README.md` (user guide).

**Available skills:**
- `agent-generator` - Create and refine specialist agents through triangulated critique

## Folder Conventions

| Folder | Purpose |
|--------|---------|
| `papers/` | Domain literature (PDFs + index) - agents reference for domain knowledge |
| `agents/` | Generated agent configs - each agent gets a subfolder |
| `src/` | All source code |
| `figures/` | Output figures |

The agent-generator skill automatically:
- Scans `papers/` for relevant literature during agent creation
- Outputs agents to `agents/<agent-name>/config.yaml`
- Creates feedback and version tracking files

---

# PART II: PROJECT SPECIFICS

Now I wanna create novel model called TFScope to predict the TF binding specificty. You should read the document (TFScope.pdf) first. In this folder, I only wanna construct the seed model to predict the initial PWM. This is a Nature series journals like Nature Methods style project targeting novel contributions in machine learning models for pwm prediction.
