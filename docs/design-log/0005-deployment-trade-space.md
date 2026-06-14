# Deployment trade-space

Long-form deep-dive for ADR-0009.

## Sub-decisions

### Development workflow — trunk-based vs feature branches

**Chosen:** trunk-based. All work merges to main daily; incomplete features behind flags.
**Rejected:** *Feature branches* — accumulate merge debt; delay eval feedback; promote big-bang PRs.
**Rationale:** with eval-as-CI, trunk-based is the only model where prompt/model regressions get caught before they reach production users. A 3-day-old feature branch can accumulate hidden eval debt.

### CI system — GitHub Actions

**Chosen:** GitHub Actions.
**Rejected:** GitLab CI (not using GitLab), Buildkite (overkill), Circle CI (no clear advantage).
**Rationale:** default for a GitHub-hosted repo, widely understood, good k8s deploy action ecosystem.

### Runtime — Kubernetes

**Chosen:** k8s. Phase 1 on localhost (Docker Compose), Phase 2 cloud k8s.
**Rejected for Phase 1:** *Cloud k8s from day one* — adds infra setup time before any code is validated.
**Rejected for Phase 2:** *Fly.io, Railway, Render* — valid alternatives; rejected because k8s was explicitly chosen. They'd be better at v1 scale operationally, but the portfolio goal includes demonstrating k8s-native deployment.
**Rationale:** Phase 1 Docker Compose → Phase 2 k8s is the right progression: build fast locally, graduate to the target runtime when the system is stable.

### Frontend — Vite + React vs Next.js

**Chosen:** Vite + React + AI SDK streaming hooks.
**Rejected:** *Next.js* — adds SSR, filesystem routing, API routes, build pipeline complexity. None of that is needed for a streaming chat UI with no SEO requirement and no public pages. A chat surface is a single-page app.
**Rationale:** the simplest thing that works. The AI SDK's `useChat` hook gives streaming out of the box; Vite gives fast dev iteration.

### Prompt versioning

**Chosen:** git files + `model_config.yaml` active-version pointer.
**Rejected:**
- *Dedicated prompt management tool (PromptLayer, Langsmith, etc.)* — adds a dependency and a UI for non-engineers to edit prompts. For v1, all prompt authors are engineers.
- *Prompts embedded in code strings* — untestable, unversioned, untrackable.
- *Prompts in a database* — runtime DB hop per turn; hard to diff; hard to PR-review.
**Rationale:** git is the right tool for versioned text artifacts that need code review and CI gates.

### A/B framework

**Chosen:** same `candidate_id` hash mechanism as canary, with named experiments in `model_config.yaml`. Logged per turn.
**Rejected:**
- *Dedicated A/B platform (LaunchDarkly, etc.)* — adds an SDK, a service dependency, and a new admin surface.
- *Random per-turn assignment* — breaks within-session consistency (a candidate gets different experiences mid-session).
**Rationale:** hash-by-candidate gives stable assignment cheaply. Named experiments in config are auditable and PR-reviewable.

### Canary auto-rollback policy

**Chosen:** 5% canary for 30 min; rollback on error rate >2× baseline or eval-key metrics regress >5%.
**Rejected:**
- *Manual rollback only* — too slow when a prompt regression causes immediate quality drop.
- *No canary (full rollout)* — no safety net.
- *1% canary* — too little traffic to get statistically useful signal in 30 min.
**Rationale:** 5%/30min gives enough signal to catch the obvious issues while bounding blast radius.

### Database migrations

**Chosen:** Alembic, forward-only in staging/prod, two-phase for destructive changes.
**Rejected:**
- *Down-migrations in prod* — operationally dangerous; encourages cutting corners.
- *Manual SQL migrations* — not reproducible, not reviewable.
**Rationale:** forward-only forces careful migration design and aligns with trunk-based dev (the code and the schema are always co-compatible going forward).

### Ingress / load balancing

**Chosen:** Nginx ingress controller (k8s standard); TLS via cert-manager + Let's Encrypt.
**Rejected:** Traefik (also fine, just less universal), cloud-native load balancers (vendor lock).
**Rationale:** standard ecosystem; works identically across GKE/EKS/AKS.

### Secrets management (Phase 2)

**Chosen:** k8s Secrets + external-secrets-operator syncing from a cloud-native secrets manager. Dev: 1Password CLI or Doppler.
**Rejected:**
- *Secrets in ConfigMaps* — unencrypted.
- *Secrets baked into images* — catastrophically bad.
- *Manual k8s Secret creation* — not IaC; not reproducible.
**Rationale:** external-secrets-operator is the k8s-native pattern; syncs from whatever secret manager you already use.

### Monitoring (deployment-relevant pieces)

**Chosen:** Prometheus + Grafana (cluster metrics, deployment health, canary error rates). Sentry (application exceptions).
**Rationale:** standard k8s observability stack. See ADR-0010 for the LLM-specific observability layer (Langfuse).
