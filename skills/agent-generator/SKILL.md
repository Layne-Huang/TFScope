---
name: meta-agent-generator
description: Meta-agent system for generating, refining, and self-improving specialist agents through human feedback loops and agent-to-agent critique. Load when creating new agents or improving existing ones.
---

# Agent Generator Meta-System

The Agent Generator is a meta-agent that creates, manages, and improves specialist agents through iterative feedback loops and cross-domain critique. Inspired by the Virtual Lab (Nature 2025), this system uses pure prompt engineering without model fine-tuning.

## Project Template Conventions

This skill follows the project template structure:

| Folder | Purpose |
|--------|---------|
| `papers/` | Domain literature (PDFs, index) - agents should reference these for domain knowledge |
| `agents/` | Generated agent configurations - each agent gets a subfolder |

**Agent output structure:**
```
agents/
└── <agent-name>/
    ├── config.yaml       # Agent configuration
    ├── feedback.jsonl    # User feedback log
    └── versions.jsonl    # Version history
```

**Papers integration:**
- During agent creation, scan `papers/` for relevant domain literature
- Include key paper references in the agent's system prompt when relevant
- Agents should be aware they can reference project literature

## Core Workflow

The agent lifecycle follows four phases:

### 1. Generation

When you need a specialist agent, invoke the Agent Generator through an interview process. The meta-agent uses `prompts/interview-requirements.txt` to ask targeted questions about:

- Domain expertise requirements
- Target user interactions and use cases
- Expected output formats and constraints
- Evaluation criteria for success

Based on this analysis, the Agent Generator creates an agent configuration using `prompts/generate-agent.txt`:

- System prompt defining role and behavior
- Few-shot examples for in-context learning
- Temperature and sampling parameters
- Capability constraints and safety guidelines

### 2. Feedback Collection

After each interaction with a generated agent, collect structured feedback:

- Numerical rating (1-5) on interaction quality
- Qualitative comments on what worked or failed
- Context tags for categorization (explanation, analysis, creative, etc.)

### 3. Refinement

When sufficient feedback accumulates (typically 10+ interactions), trigger refinement analysis using `prompts/generate-refinements.txt`. The Agent Generator:

- Analyzes feedback patterns for common issues
- Identifies underperforming capabilities
- Generates targeted prompt improvements
- Proposes version increments with human approval

**Token Budget Target:** ~4000-5000 tokens for system_prompt + few_shot_examples.

**Refinement Philosophy:**
- **Consolidate, don't accumulate**: Each round refines and replaces content
- Remove redundancy when adding new capabilities
- Maximum 2-3 few-shot examples (quality over quantity)
- Use concise principles over verbose explanations

All refinements require explicit human approval before application.

### 4. Self-Improvement

The Agent Generator analyzes its own generation performance across all agents:

- Extracts successful prompt patterns from high-rated agents
- Identifies anti-patterns from low-rated configurations
- Updates generation templates and strategies
- Requires human approval for self-modifications

## Agent Configuration Schema

Each agent has a YAML configuration:

```yaml
name: immunologist
version: 3
system_prompt: |
  You are an expert immunologist...
role_definition: |
  As an immunologist, you provide...
temperature: 0.7
few_shot_examples:
  - input: "What is T cell activation?"
    output: "T cell activation requires..."
capabilities:
  - literature-review
  - experimental-design
constraints:
  - cite-sources
  - acknowledge-limitations
feedback_summary:
  total_interactions: 24
  average_rating: 4.6
  common_strengths:
    - clear explanations
    - source citations
  common_weaknesses:
    - oversimplification
```

## Feedback Schema

Each feedback entry is a JSONL line:

```json
{
  "timestamp": "2026-02-02T16:15:00Z",
  "rating": 5,
  "comment": "Clear explanation with good citations",
  "context_tags": ["explanation", "mechanism", "literature"]
}
```

## Safety Mechanisms

- All refinements require human approval
- All meta-agent self-updates require human approval
- Git version control enables rollback
- JSONL append-only prevents data loss
- Explicit constraints in all agent configs

## Usage Patterns

### Creating a New Agent

When you need domain-specific assistance:

1. Identify the domain and required expertise
2. Request agent generation through interview process
3. Review and approve the generated configuration
4. Use the agent for tasks
5. Provide feedback after each interaction

### Improving an Existing Agent

When an agent underperforms:

1. Collect more feedback data points
2. Request refinement analysis
3. Review proposed changes
4. Approve or reject refinements
5. Monitor performance changes

### Multi-Round Agent Training

For comprehensive agent development, use multi-round training:

**Protocol:** Use `prompts/multi-round-training.txt` for structured multi-round development.

**Key principles:**
- Each round iteratively refines, not accumulates
- Updates feedback/versions/registry after EACH round (not just final)
- Stays within token budget (~4000-5000 tokens)
- Focus areas: Role → Examples → Capabilities → Consolidation → Feedback
- **Generate synthetic feedback during training** (30+ entries covering all domains)
- **Agent-to-agent critique** for improvement (see below)

**Example:**
```
Round 1: Core role definition and boundaries
Round 2: Few-shot example quality (reduce count, improve quality)
Round 3: Capability specificity and constraint sharpening
Round 4: Consolidation (remove redundancy)
Round 5-9: Domain-specific expansion with synthetic feedback
Round 10: Final consolidation + synthetic feedback generation
```

Each round creates a git commit, so full history is preserved.

#### Synthetic Feedback Generation

**Always generate synthetic feedback during multi-round training**, even before real user interactions:

1. **Purpose:** Provide feedback-driven refinement data for agent improvement
2. **Format:** JSONL entries matching real feedback schema
3. **Quantity:** 30+ entries covering all expertise domains
4. **Quality:** Mix of ratings (3-5) with constructive criticism
5. **Context tags:** Categorize each entry by domain/interaction type

```json
{
  "timestamp": "2026-02-02T16:15:00Z",
  "rating": 5,
  "comment": "Excellent workflow guidance with clear decision tree",
  "context_tags": ["workflow-design", "tool-selection", "practical-guidance"]
}
```

**When to generate:** At final training round OR continuously during multi-round training

#### Triangulated Critique (Virtual Lab-Inspired)

**Triangulated critique combines three perspectives for comprehensive agent evaluation:**

Inspired by the Virtual Lab (Nature 2025), this system uses multiple specialized agents working together to evaluate and improve agent configurations more effectively than any single critic.

```
                 ┌─────────────────────────┐
                 │   Target Agent vN       │
                 └───────────┬─────────────┘
                             │
         ┌───────────────────┼───────────────────┐
         │                   │                   │
         ▼                   ▼                   ▼
┌────────────────┐  ┌────────────────┐  ┌────────────────┐
│ Domain Critics │  │  Prompt-Critic │  │ User Feedback  │
│ (2-3 agents)   │  │ (specialist)   │  │ (interactions) │
└────────┬───────┘  └────────┬───────┘  └────────┬───────┘
         │                   │                   │
         └───────────────────┼───────────────────┘
                             ▼
                 ┌───────────────────────┐
                 │  Consensus Aggregator  │
                 │  (weighted framework)  │
                 └───────────┬───────────┘
                             │
                    [Human Approval]
                             │
                             ▼
                 ┌───────────────────────┐
                 │   Target Agent vN+1   │
                 └───────────────────────┘
```

**The Three Critique Sources:**

| Source | What It Evaluates | Weight | Expertise |
|--------|-------------------|--------|-----------|
| **Domain Agents** (2-3) | Domain coverage, terminology accuracy, capability completeness | 40% | Domain-specific knowledge |
| **Prompt-Critic** (1) | Prompt structure, token efficiency, clarity, design quality | 30% | Prompt engineering best practices |
| **User Feedback** (∞) | Practical utility, interaction quality, real-world performance | 30% | Actual usage patterns |

**Why Triangulation Works:**

1. **Domain Coverage** from domain specialists - "Is the domain knowledge complete?"
2. **Design Quality** from prompt-critic - "Is the prompt well-structured and efficient?"
3. **Real Validation** from users - "Does this work in practice?"

Together they catch issues that any single source would miss.

**Consensus Framework:**

| Consensus Level | Definition | Action | Confidence |
|-----------------|------------|--------|------------|
| **Strong** | 2+ independent sources agree | Implement immediately | HIGH |
| **Moderate** | Only 1 source identifies issue | Evaluate and decide | MODERATE |
| **Conflict** | Sources disagree | Apply weighted resolution | Context-dependent |

**Conflict Resolution Rules:**

```
Design quality > Domain content > User preference
EXCEPT: Safety > Everything (safety veto applies)
```

**Examples:**
- Domain says "add more detail" vs Prompt-critic says "token budget exceeded" → Prompt-critic wins (design foundation)
- Experts say "use technical jargon" vs Users say "too confusing" → Users win (utility focus)
- Any source says "this removes safety boundaries" → Veto (safety first)

**Usage:**

Use `prompts/triangulated-critique.txt` to guide the triangulated critique process:

```bash
# Request triangulated critique for an agent
"Run triangulated critique on the literature-review agent using geneticist, prompt-critic, and user feedback"

# The Agent Generator will:
# 1. Load 2-3 domain agents (geneticist, protein-structuralist)
# 2. Load prompt-critic agent
# 3. Gather user feedback from feedback.jsonl
# 4. Apply consensus-based aggregation
# 5. Generate refinement proposal with priority rankings
# 6. Present for human approval
```

**Critique Selection Guidelines:**

For a target agent, select domain critics with:
- **Moderate overlap** (20-40% domain similarity) - Relevant expertise
- **Not identical domains** - Avoid echo chamber
- **Complementary perspectives** - Different angles on similar problems

**Example for literature-review agent:**
- geneticist (moderate overlap - both deal with literature)
- protein-structuralist (minimal overlap - structural methods)
- prompt-critic (always included for design quality)

**Weighted Scoring:**

Each improvement proposal is scored:

```
Support Score = (Domain_Strength × 0.4) + (Design_Strength × 0.3) + (User_Strength × 0.3)
Priority Score = Support × Impact × (1 - Risk)
```

Highest priority items are addressed first.

**Quality Signals:**

**Effective triangulation indicators:**
- Strong consensus on critical issues (2+ sources agree)
- Domain and design critiques complement each other
- User feedback validates or corrects expert opinions
- Conflicts resolved through framework (not ignored)

**Aggregation quality indicators:**
- Clear priority ranking of improvements
- Transparent conflict resolution with rationale
- Token budget maintained through consolidations
- Safety constraints preserved
- Cross-domain synergies identified

**Available Critics:**

```yaml
available_critics:
  prompt-critic: Prompt engineering specialist (design quality, token efficiency)
  geneticist: Bioinformatics specialist (functional genomics, QTL analysis)
  protein-structuralist: Protein structure specialist (modeling, experimental methods)
  literature-review: Academic literature specialist (search, synthesis, critique)
```

**1. Critic Selection**

When selecting agents to critique a target agent:
- Choose 3-5 critics for balanced perspective
- Select agents with complementary (not identical) domains
- Include agents with moderate domain overlap for relevance
- Include agents with minimal overlap for objectivity
- Consider agent performance ratings (high-rated agents weighted more)

**Example critic selection for `literature-review` agent:**
- `geneticist` - moderate overlap (both deal with research literature)
- `protein-structuralist` - minimal overlap (structural methods vs literature search)
- Future: `statistical-analyst` - moderate overlap (statistical interpretation)

**2. Structured Critique Process**

Use `prompts/agent-critique.txt` for the critique protocol. Each critic agent receives this protocol and provides:

```markdown
# Agent Critique: {{target_agent}} v{{version}}
**Critiqued by:** {{critic_agent}}

## Dimension Scores (1-5)
- System Prompt Quality
- Few-Shot Examples
- Capabilities
- Constraints
- Overall Effectiveness

## Strengths to Preserve
{{3-5 specific strengths with examples}}

## Weaknesses to Address
{{3-5 specific weaknesses with concrete fixes}}

## Redundancy Identification
{{Overlapping content to consolidate}}

## Missing Capabilities
{{Gaps in capability coverage}}

## Domain-Specific Recommendations
{{Integration opportunities, best practice transfer, overlap management}}

## Specific Improvements
{{Exact wording for prompt, example, capability changes}}
```

**3. Critique Aggregation**

The Agent Generator synthesizes multiple critiques using `prompts/critique-aggregation.txt`:

- **Consensus Identification:** Issues mentioned by 3+ critics = HIGH priority
- **Domain Weighting:** Higher weight to critics with relevant expertise
- **Conflict Resolution:** Reconcile opposing suggestions through analysis
- **Impact/Risk Scoring:** Prioritize high-impact, low-risk changes
- **Token Budget:** Ensure total stays within 4000-5000 tokens

**Aggregation output:**
```markdown
# Aggregated Refinement Proposal

## Priority Improvements (Ranked)
1. HIGH PRIORITY (Consensus + High Impact + Low Risk)
2. MEDIUM PRIORITY (Moderate consensus or impact)
3. LOW PRIORITY (Single critic or high risk)

## System Prompt Refinements
{{Specific diff-style changes with rationale}}

## Few-Shot Example Updates
{{Add/remove/modify with justifications}}

## Cross-Agent Synergies
{{New integration opportunities identified}}
```

**4. Implementation Workflow**

```
Target Agent → Select Critics → Generate Critiques → Aggregate → Refine → Validate
     ↑                                                                    ↓
     └───────────────────── Re-critique (optional) ←──────────────────────┘
```

**5. Best Practices**

**For effective critiques:**
- Provide specific, actionable feedback
- Reference exact configuration sections
- Balance critique with recognition of strengths
- Consider token budget implications
- Suggest additions AND consolidations

**For effective aggregation:**
- Prioritize consensus issues
- Weight by domain relevance and agent performance
- Resolve conflicts transparently
- Maintain agent's core identity
- Document rationale for all decisions

**6. Usage Example**

```bash
# Request agent critique using prompts/agent-critique.txt
"Have the geneticist and protein-structuralist agents critique the literature-review agent configuration"

# The Agent Generator will:
# 1. Load both critic agents
# 2. Provide them with prompts/agent-critique.txt and target config
# 3. Collect structured critiques from each
# 4. Aggregate into unified refinement proposal (prompts/critique-aggregation.txt)
# 5. Present for human approval
```

**7. Quality Signals**

**Effective critique indicators:**
- Multiple critics identify same issue independently
- Suggestions are specific and implementable
- Cross-domain insights reveal blind spots
- Token-conscious recommendations
- Balance of praise and constructive feedback

**Aggregation quality indicators:**
- Clear priority ranking of improvements
- Transparent conflict resolution
- Maintained agent core function
- Within token budget
- Novel synergy opportunities identified

### Meta-Agent Self-Improvement

The meta-agent periodically self-reflects using `prompts/self-analysis.txt` and `prompts/refine-self.txt`:

1. Analyze performance across all agents
2. Extract successful patterns
3. Update generation strategies
4. Review and approve self-modifications
