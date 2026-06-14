Status: completed

## What to build

GitHub Actions CI/CD pipeline (trunk-based development) and Kubernetes manifests for Phase 2. CI runs on every PR: lint (Ruff, Biome), type check (mypy, tsc), unit tests, integration tests, and path-gated eval slice. On merge to main: build Docker images, push to registry, deploy to staging. On tagged release: deploy to prod with 5% canary + auto-rollback.

k8s manifests cover all five services (gateway, agent, corpus, ingestion, math-verifier), Postgres, Qdrant, Langfuse, and the TEI embedder server. Written and validated (`kubectl --dry-run=client`) in Phase 1 so they're tested before cloud infra is provisioned.

## Acceptance criteria

- [ ] `ci.yml`: lint + type check + unit tests run on every PR; fail fast
- [ ] `ci.yml`: integration tests run on every PR (use Docker Compose services in GH Actions)
- [ ] `ci.yml`: eval slice runs only on PRs touching `prompts/`, `services/agent/`, `services/gateway/`, `evals/`, `config/` (path filter)
- [ ] `deploy.yml`: on merge to main, Docker images built and pushed; staging deploy triggered
- [ ] `deploy.yml`: on release tag, prod canary deploy (5%) with auto-rollback on error spike
- [ ] Each service has a `Dockerfile` (multi-stage, non-root user, minimal image)
- [ ] `infra/k8s/` contains Deployment, Service, and ConfigMap manifests for all services
- [ ] `kubectl apply --dry-run=client -f infra/k8s/` passes with no errors
- [ ] Secrets referenced as k8s Secret refs (not hardcoded in manifests)
- [ ] Health check endpoints: `GET /healthz` on each FastAPI service returns 200

## Blocked by

`.scratch/gre-tutor-v1/issues/11-eval-harness.md`
