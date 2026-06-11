# Agent Generator User Guide

The Agent Generator is a meta-agent system that creates and improves specialist agents through **triangulated critique**—combining domain agent feedback, prompt engineering review, and real-world user feedback.

Inspired by the Virtual Lab (Nature 2025), this system uses pure prompt engineering without model fine-tuning to continuously improve agent quality.

## Project Template Integration

This skill follows the project template conventions:

| Folder | Purpose |
|--------|---------|
| `papers/` | Domain literature - agents reference these for domain knowledge |
| `agents/` | Generated agents - each gets a subfolder with config, feedback, versions |

**Generated agent structure:**
```
agents/<agent-name>/
├── config.yaml       # Agent configuration
├── feedback.jsonl    # User feedback log
└── versions.jsonl    # Version history
```

During agent creation, the generator will:
1. Scan `papers/` for relevant domain literature
2. Ask which papers the agent should reference
3. Include paper awareness in the agent's system prompt

## Quick Start

### Creating Your First Agent

To create a specialist agent, specify the domain you need:

"Create an immunologist agent for explaining immune mechanisms and suggesting experiments."

The Agent Generator will conduct an interview to understand:

- What specific immunology topics you work with
- What level of detail you prefer
- What types of tasks you'll perform (literature review, experimental design, data analysis)
- What output formats work best for you

After the interview, it generates an agent configuration for your approval.

### How Agents Improve

Once created, agents improve through **triangulated critique**:

```
Target Agent → Domain Critics + Prompt-Critic + User Feedback → Consensus → Refinement
```

| Source | What It Evaluates | Examples |
|--------|-------------------|----------|
| **Domain Agents** (2-3) | Domain coverage, terminology, capabilities | geneticist reviewing literature-review for genetics-specific capabilities |
| **Prompt-Critic** (1) | Prompt structure, token efficiency, clarity | Identifying redundancies, suggesting consolidations |
| **User Feedback** (∞) | Real-world performance, utility | Rating interactions, noting what worked/failed |

**Only act on strong consensus** (2+ sources agree) for major changes.

## Workflow Examples

### Iterative Improvement with Triangulated Critique

**Scenario:** literature-review agent v3 has 3.8/5.0 average rating

**Domain Critics Say:**
- geneticist: "Missing advanced search strategies for variant databases"
- protein-structuralist: "No coverage of structural biology literature"

**Prompt-Critic Says:**
- "Few-shot examples too verbose (~800 tokens each, target 300-500)"
- "Missing out-of-scope boundaries for medical advice"

**User Feedback Shows:**
- 4/5: "Great summaries, but can't find preprints efficiently"
- 3/5: "Responses are long and detailed but sometimes miss key sources"

**Consensus Analysis:**
- **Strong consensus:** Verbose examples (Prompt-critic + users agree)
- **Strong consensus:** Missing preprint guidance (geneticist + users)
- **Moderate consensus:** Structural literature gap (only protein-structuralist)

**Refinement Applied:**
- Condense few-shot examples by 40% (-300 tokens)
- Add bioRxiv/arXiv search strategies
- Add "out-of-scope: medical advice" boundary
- Add structural literature coverage in future round

**Result:** v4 achieves 4.6/5.0 average rating

### Creating New Agents

**Request:** "Create a molecular biologist agent for CRISPR experimental design."

**Interview Questions:**
- What CRISPR systems do you work with? (SpCas9, base editing, prime editing?)
- What organisms? (Mammalian cells, bacteria, plants?)
- What deliverables? (gRNA sequences, donor templates, protocols?)

**Generated Agent:**
- Specialized in your CRISPR system
- Tailored to your organism
- Outputs in your preferred format

**After Creation:**
- Use agent for tasks
- Provide feedback after each interaction
- After 10+ feedback entries, request triangulated critique

### Multi-Agent Collaboration

Agents can work together on complex tasks:

```
Research Question → literature-review (find papers)
                    ↓
                 geneticist (analyze variant data)
                    ↓
         protein-structuralist (model protein impact)
                    ↓
             prompt-critic (critique any agent's performance)
```

## Triangulated Critique System

### How It Works

1. **Collect Perspectives:**
   - Domain agents review domain coverage
   - Prompt-critic reviews design quality
   - User feedback shows real-world performance

2. **Build Consensus:**
   - Strong consensus (2+ sources agree): ACT immediately
   - Moderate consensus (1 source): EVALUATE
   - Conflicting opinions: RESOLVE through framework

3. **Generate Refinement:**
   - Prioritize by support × impact × (1 - risk)
   - Address high-priority items first
   - Maintain token budget (4000-5000 tokens)

4. **Human Approval:**
   - Review proposed changes
   - Approve, modify, or reject
   - Apply and increment version

### Conflict Resolution

When sources disagree:

| Conflict Type | Resolution |
|---------------|------------|
| Domain vs Design | Design quality takes precedence (foundation) |
| User vs Experts | User preference wins (utility focus) |
| Expert A vs Expert B | Prompt-critic breaks tie (design principles) |
| **Any vs Safety** | **Safety always wins** (veto power) |

### Available Critics

```yaml
Domain Agents:
  - geneticist: Bioinformatics specialist
  - protein-structuralist: Protein structure specialist
  - literature-review: Academic literature specialist

Meta-Critic:
  - prompt-critic: Prompt engineering specialist (domain-agnostic)
```

## Best Practices

1. **Provide Specific Feedback:** "Missing X capability" is more helpful than "not good enough"

2. **Wait for Consensus:** Major changes require 2+ sources agreeing

3. **Trust User Feedback:** Practical utility matters most

4. **Let Prompt-Critic Optimize:** Token efficiency and structure are their expertise

5. **Domain Experts Know Content:** Trust domain agents on what should be covered

## Understanding Agent Quality

### What Makes a Good Agent?

**System Prompt (40% of quality):**
- Clear role definition
- Specific responsibilities
- Explicit boundaries (out-of-scope)
- Appropriate tone for domain
- Concise yet complete (~3000 tokens)

**Few-Shot Examples (30% of quality):**
- Demonstrate ideal behavior
- Cover diverse scenarios
- Token-efficient (300-500 each)
- 2-4 maximum (quality over quantity)

**Capabilities (15% of quality):**
- Complete domain coverage
- Action-oriented naming
- Appropriate granularity
- No redundancy

**Constraints (15% of quality):**
- Safety boundaries included
- Enforceable through prompt
- Not over-constrained

### Token Budget

**Target:** 4000-5000 tokens (system_prompt + few_shot_examples)

**Why:** Fits in context window, leaves room for user query, enables fast responses

**If over budget:**
- Consolidate redundant content
- Replace verbose explanations with principles
- Remove or merge overlapping examples

## Advanced Features

### Multi-Round Training

For comprehensive agent development, use 10-round training:

1. **Round 1:** Core role definition
2. **Round 2:** Few-shot example refinement
3. **Round 3:** Capability specification
4. **Round 4:** Consolidation
5. **Rounds 5-9:** Domain expansion with synthetic feedback
6. **Round 10:** Final consolidation

Each round creates a git commit for full history.

### Version Control

Every agent change is tracked in `versions.jsonl`:

```json
{
  "version": 3,
  "timestamp": "2026-02-02T16:15:00Z",
  "changes": "Added preprint search, consolidated examples",
  "trigger": "triangulated_critique"
}
```

This enables rollback if a refinement degrades performance.

### Pattern Selection

The Agent Generator has pre-built patterns:

- **Technical Domain:** Deep expertise, literature-focused
- **Creative Writing:** Emphasis on style, narrative, voice
- **Data Analysis:** Statistical reasoning, visualization

## Troubleshooting

### Agent Performance Plateaus

If an agent stops improving:

- Check if feedback is consistent (conflicting feedback confuses analysis)
- Request triangulated critique to identify blind spots
- Consider if domain has evolved (new papers, new techniques)

### Low User Ratings

If user feedback shows < 4.0 average rating:

- **Immediate:** Request prompt-critic review (often finds structural issues)
- **Secondary:** Request domain agent critique (identifies content gaps)
- **Tertiary:** Analyze user feedback themes directly

### Token Budget Issues

If responses are too long/verbose:

- Prompt-critic identifies consolidation opportunities
- Remove redundant content before adding new features
- Replace examples with principles where appropriate

### Conflict Between Critics

If domain agents and prompt-critic disagree:

- Design quality wins over domain preference (foundation first)
- But user preference wins over expert opinion (utility focus)
- Safety veto applies to any change removing boundaries

## Architecture Notes

This system uses **prompt engineering only**. No model fine-tuning occurs. All improvements come from:

- Better system prompts (clearer, more complete)
- Better few-shot examples (higher quality, better coverage)
- Refined constraints (appropriate, enforceable)
- Optimized token usage (consolidated, efficient)

This approach is:
- **Safer:** All changes are transparent and reviewable
- **Debuggable:** Can trace any issue to specific prompt changes
- **Reversible:** Git history enables rollback
- **Transferable:** Patterns work across domains

## Feedback Schema

Each feedback entry is stored in `agents/<agent-name>/feedback.jsonl`:

```json
{
  "timestamp": "2026-02-02T16:15:00Z",
  "rating": 5,
  "comment": "Clear explanation with good citations",
  "context_tags": ["explanation", "mechanism", "literature"]
}
```

**Fields:**
- `timestamp`: ISO 8601 UTC timestamp
- `rating`: Integer 1-5 (5=excellent, 1=poor)
- `comment`: Optional qualitative feedback
- `context_tags`: Array categorizing interaction type

**Trigger refinement when:**
- 10+ feedback entries accumulated
- Average rating drops below 4.0
- User explicitly requests critique
