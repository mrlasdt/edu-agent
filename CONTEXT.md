# Edu Agent

An LLM-powered GRE test-prep tutor that helps Candidates improve their score on the GRE General Test. v1 is English-only and section-scoped (see below); the system is architected for multilingual extension and additional test types in v2. Deployed as a web chat.

## Language

**Tutor mode**:
The agent's default interaction mode. In **Basic** complexity, the agent scaffolds the Candidate toward an answer through hints and Socratic questions without revealing it. In **Advanced** complexity, the agent emits a **plan** — outline, study sequence, or essay structure — rather than the worked artifact, so the Candidate produces the artifact themselves.
_Avoid_: Teaching mode, guide mode, hint mode

**Solve mode**:
The interaction mode in which the agent reveals the fully worked artifact: in **Basic**, a worked solution (steps + explanation) for a Quant problem or a worked sample essay for an AW prompt; in **Advanced**, the full multi-step study plan or a complete essay. Entered only by deliberate Candidate action. Bare-answer-only output is never produced.
_Avoid_: Answer mode, do-it-for-me mode, full-solve

**Mode switch**:
The Candidate-initiated action that moves the conversation from Tutor mode into Solve mode. In Basic, resets to Tutor at the start of each new question. In Advanced, the unit of reset is the task (essay or study plan), not the turn.
_Avoid_: Mode toggle, override, reveal

**Basic complexity**:
The default session complexity. One question or one essay prompt at a time, single-step turns, routed through a section-specific agent (Quant Agent or AW Agent). Tutor or Solve mode applies per question.
_Avoid_: Standard mode, quick mode

**Advanced complexity**:
A session complexity intended for multi-step tasks: a full study plan over weeks, a multi-pass essay revision cycle, or a diagnostic-and-targeted-practice flow. Uses a planner-executor pattern with specialist sub-agents and an assembler. Architected in v1 but not implemented; v2.
_Avoid_: Project mode, plan mode, expert mode

## Roles

**Candidate**:
The end user of the agent — a GRE test-taker preparing for the General Test. Adult, English-proficient by definition (GRE is administered in English). May be a native English speaker or not. v1 makes no distinction between sub-populations.
_Avoid_: Student, user, learner, test-taker

**Teacher**:
A user who uploads study material into a specific School corpus (e.g. a prep-course instructor with their own curated drill sets and model essays). Cannot tutor Candidates through the agent; their only authority is over content. School / Teacher tier is architected but not user-facing in v1.
_Avoid_: Instructor, coach, tutor

**Admin**:
A user who curates and writes to the Global corpus (ETS sample essays at each score tier, official practice questions, the published AW rubric, etc.). Single privileged role; not a per-school superuser.
_Avoid_: Owner, root, superuser

**School**:
The tenant boundary for Teacher-uploaded content (e.g. a prep-course business). Every Teacher and every Candidate belongs to exactly one School. Determines which School corpus the Candidate's retrieval can see. School tier is architected for v1, user-facing in v2.
_Avoid_: Organisation, tenant, course

## Sections

**Quant**:
The GRE Quantitative Reasoning section. Question types: Quantitative Comparison (QC), Multiple Choice (single answer), Multiple Choice (one or more answers), Numeric Entry, and Data Interpretation sets. Score 130–170 in 1-point increments.
_Avoid_: Math, quantitative, maths

**AW**:
The GRE Analytical Writing section. Two task types: Issue task (write an essay analysing an issue) and Argument task (analyse and critique an argument). Score 0–6 in 0.5-point increments per task, averaged.
_Avoid_: Writing, essay, analytical writing (full form)

**Verbal**:
The GRE Verbal Reasoning section. Question types: Text Completion, Sentence Equivalence, Reading Comprehension. Score 130–170. **Out of scope for v1**; defer to v2.
_Avoid_: Reading, vocabulary, verbal reasoning (full form)

**Issue task**:
The first of the two AW tasks. The Candidate is given a claim and asked to take a position and develop a reasoned argument. Scored on the published ETS rubric.
_Avoid_: Issue prompt, opinion essay

**Argument task**:
The second of the two AW tasks. The Candidate is given a short argument and asked to critique its reasoning. Scored on the same published ETS rubric.
_Avoid_: Critique essay, argument analysis

## Corpus

The agent retrieves from three tiers, merged with provenance preserved end-to-end through to citation.

**Global corpus**:
The shared, curated corpus written only by Admins. Sourced from publicly published ETS materials: official practice questions, sample AW essays at each scoring tier (1 through 6), the published AW rubric, official strategy guides. Visible to every Candidate.
_Avoid_: Base corpus, public corpus, root corpus

**School corpus**:
Per-tenant corpus written by Teachers belonging to that School (a prep-course's curated drill sets, model essays, internal notes). Visible only to Candidates in the same School. Higher retrieval priority than the Global corpus when a relevant chunk is available. Architected in v1, user-facing in v2.
_Avoid_: Teacher corpus, course corpus

**Candidate corpus**:
Per-candidate private corpus written by the Candidate themselves (their own essay drafts, study notes, error logs). Visible only to that Candidate. Highest retrieval priority.
_Avoid_: Personal corpus, my notes, private upload

**Citation tier**:
The corpus tier that a quoted passage came from, surfaced in the output ("from official ETS materials" / "from your course notes" / "from your own notes"). Required on every AW or Verbal claim that references source material.
_Avoid_: Source label, attribution

## Style

**Canonical style**:
The default AW-output style — an essay written to score in the 5–6 range on the published ETS Analytical Writing rubric. Concretely: clear thesis, well-developed positions, sustained logical flow, appropriate use of evidence and examples, varied syntax, near-perfect mechanics. Anchored by the public sample essays in the Global corpus.
_Avoid_: Default style, ETS style, high-score style

**Personal style**:
The optional, opt-in style variant in which the agent biases voice and vocabulary toward the Candidate's own prior writing while keeping the Canonical style's structural rubric intact. Active only when the Candidate has opted in AND has at least two essays in their Candidate corpus.
_Avoid_: Student style, custom style, individual style, voice match

**Style profile**:
A per-Candidate JSON record derived asynchronously from the Candidate corpus, capturing structural and lexical biases (sentence length distribution, transition vocabulary, paragraph density, etc.). Not the essays themselves; a summary used to augment generation prompts when Personal style is active.
_Avoid_: Voice profile, writing profile, stylometric profile

## Architecture

**Orchestrator**:
The logic layer inside the Gateway service that every turn passes through before reaching any agent. Performs four steps in order: validate (rules), enrich (Haiku-tier LLM — detects incomplete questions), complexity-classify (Haiku-tier LLM — detects Basic-vs-Advanced mismatch), route. Resolves the target agent via the Agent Registry. Designed as an extension point: in v1 it routes to Basic only; the Advanced router is stubbed.
_Avoid_: Router, gateway, dispatcher, controller

**Agent registry**:
A plugin registry inside the Agent service. Agents register by `(test_type, section)` key; the Orchestrator resolves the correct agent class by looking up the session's `(test_type, section)`. The primary extension seam for adding new test types (IELTS, GMAT, TOEFL) without changing the Orchestrator or other services.
_Avoid_: Agent factory, agent map, agent pool

**Test type**:
The standardised test a Candidate is preparing for. A first-class field on session state and corpus document metadata from v1. v1 supports `gre` only; v2 adds further values. Determines which agents, prompts, corpus documents, and golden set items apply to a session.
_Avoid_: Exam type, test, product type

**Clarification turn**:
A turn in which the Orchestrator detects the Candidate's question is incomplete (e.g. "solve this" with no problem attached, or an essay request without specifying Issue or Argument) and asks for missing context, instead of forwarding to an agent. Counts as a turn in conversation history but does not invoke the main agent.
_Avoid_: Follow-up, retry, prompt-for-info

**Complexity escalation**:
A suggestion the Orchestrator surfaces when a turn in Basic complexity looks like it belongs in Advanced (or vice-versa). The Candidate decides whether to switch — never silently re-routed.
_Avoid_: Auto-promote, escalate, upgrade

## Quality criteria

**Quant correctness**:
The final symbolic or numeric answer in a Quant turn agrees with the sympy-sandbox-computed answer, after at most two retries and at most one stronger-model escalation. The hard correctness criterion — any Quant turn that fails this is a `verifier-fail` event and degrades gracefully rather than emitting an unverified answer.
_Avoid_: Right answer, accuracy, correctness

**Verifier-fail**:
A Quant turn in which the math verifier and the agent could not agree on an answer after the full retry-and-escalate pipeline. The agent must visibly degrade (Tutor: "let's reason through it together"; Solve: "I'm not confident in the final answer") rather than emit an unverified claim. Tracked as an eval signal.
_Avoid_: Failed turn, math fail

**Rubric score**:
The estimated 0–6 score that a generated or Candidate-submitted AW essay would receive on the ETS rubric, as judged by an LLM judge calibrated against the public sample essays. Used both at eval time (to score canonical-style output) and at runtime (to give Candidates targeted feedback on their own essays).
_Avoid_: Essay score, score estimate

## Eval

**Golden set**:
The closed, hand-curated set of GRE items used as the offline eval baseline: ~150 Quant problems (proportional across question types i–iv) and 24 AW prompts (12 Issue + 12 Argument), all sourced from publicly released ETS materials. The single source of truth for "did this change improve or regress quality?"
_Avoid_: Eval set, test set, fixtures

**Anchor essays**:
The six published ETS sample essays per AW prompt — one at each scoring tier (1 through 6). Fed in-context to the AW judge alongside the rubric so the judge scores on the same calibrated scale across runs. Without anchors, absolute scores drift; with anchors, they don't.
_Avoid_: Sample essays, reference essays, calibration essays

**MCP server**:
A Model Context Protocol server that exposes agent capabilities to external MCP-capable clients (e.g. Claude Desktop). Two in v1: the **Candidate MCP server** (tools: `search_corpus`, `get_question`, `get_model_essay`) and the **Admin MCP server** (tools: `ingest_document`, `list_ingestion_jobs`, `get_job_status`). Both are thin wrappers around the same service layer as the web backend — no duplicate logic.
_Avoid_: Plugin, extension, integration

**Admin UI**:
A minimal web interface for Admins to manage the Global corpus: file upload (PDF, DOCX, TXT), ingestion job status table, failed-job log. Not Candidate-facing. Runs behind auth. Implemented in v1 so corpus management doesn't depend on external tooling.
_Avoid_: Admin panel, dashboard, backoffice

**Human review queue**:
A queue of eval items flagged for human review — low-judge-confidence scores, boundary AW scores (3.5 / 4.5), drift-alert investigation samples, and a 1% sample of verifier-fail events. v1 implementation is a stub (writes to a local queue table); v2 integrates the `gotohuman` service.
_Avoid_: Human-in-the-loop queue, audit queue
