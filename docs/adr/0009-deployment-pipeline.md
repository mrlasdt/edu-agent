# Deployment pipeline

The agent ships via a trunk-based development workflow on GitHub Actions, targeting Kubernetes as the production runtime, with **Phase 1 running entirely on localhost** (Docker Compose) to enable rapid iteration before cloud infrastructure is provisioned. The frontend is a lightweight React app (Vite + React + AI SDK streaming) — no Next.js. Prompts live in Git as versioned files. Environments are dev (localhost), staging (k8s cluster, non-prod), and prod (k8s cluster). Canary deploys route 5% of traffic by `candidate_id` hash; auto-rollback fires on error-rate or eval-metric regression. Database migrations are forward-only in staging/prod via Alembic.

## Status

Accepted.

## Phases

**Phase 1 — localhost:** Docker Compose orchestrates all services (gateway, agent, corpus, ingestion worker, math-verifier MCP server, Postgres, Qdrant, Langfuse, BGE-M3 TEI server). No cloud infra. Eval, guardrail, and agent pipelines fully functional against local Qdrant + Postgres. Secrets via `.env` (git-ignored). The k8s manifests are written in Phase 1 so they're tested before cloud is needed.

**Phase 2 — cloud k8s:** Apply manifests to a managed k8s cluster (GKE, EKS, or AKS — one config change). Neon or managed Postgres replaces local Postgres. Qdrant Cloud replaces local Qdrant. GitHub Actions promotes images on merge to main; canary + rollback policies activate.

## Considered options

- **Fly.io / Railway / Render** — simpler v1 ops; rejected because k8s was explicitly chosen. The Phase 1 Docker Compose approach preserves local simplicity while building the k8s-native artifacts in parallel.
- **Next.js frontend** — rejected in favour of Vite + React. Next.js adds SSR complexity that's unnecessary for a streaming chat UI with no SEO requirement.
- **Feature-branch-based development** — rejected in favour of trunk-based. Feature branches accumulate merge debt; trunk forces smaller commits and faster eval feedback.
- **Dedicated LLM-ops platform (PromptLayer, Langfuse Cloud as the CD target)** — rejected for v1 prompt CD; git-as-source-of-truth is simpler and more auditable. Langfuse is opted in for *observability* (see ADR-0010), not for prompt deployment.

## Consequences

- **Trunk-based dev means every merge to main must pass all CI gates** — lint, type-check, unit tests, integration tests, and the path-gated per-PR eval slice (ADR-0006). PRs are short-lived (≤1 day); feature flags gate incomplete features rather than long-lived branches.
- **Prompt versioning is git-native.** `prompts/{agent}/{mode}/v{N}.md` with YAML frontmatter. Active version per `(agent, mode)` set in `model_config.yaml`. Old versions retained in git; never deleted. A/B traffic splits configured in the same file: `aw_agent.tutor: { v3: 90, v4: 10 }`.
- **k8s manifests are first-class artifacts.** Written in Phase 1, validated with `kubectl --dry-run=client` in CI, applied to staging on merge to main, applied to prod on tagged release.
- **Canary auto-rollback is the load-bearing production safety feature.** 5% canary for 30 min; rollback on error rate >2× baseline OR eval key metrics regress >5%. Promotion to 100% is automatic on healthy.
- **Database migrations are Alembic, forward-only in staging/prod.** Column/table removal goes via two-phase deploy: ship code that doesn't use the column; drop in a subsequent release. No down-migrations in production.
- **Per-PR Neon preview branches** (Phase 2) give each PR an isolated Postgres for integration tests. Phase 1 uses a local Postgres container reset between runs.
- **Secrets:** Phase 1 — `.env` file (git-ignored). Phase 2 — k8s Secrets + external-secrets-operator syncing from a secrets manager (1Password CLI or Doppler for dev; cloud-native secrets manager for prod).
- **Langfuse** is opted in as the LLM observability layer (per ADR-0010). It runs locally in Phase 1 (Docker Compose service) and on a managed instance in Phase 2.
