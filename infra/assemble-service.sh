#!/bin/sh
# Usage: assemble-service.sh <service> <repo-dir> <out-dir>
# Run by infra/Dockerfile. Copies what <service> needs from <repo-dir> into <out-dir>/app and
# writes its start command (`serve`) and health probe (`healthcheck`) into <out-dir>/bin.
# The table below is the one place that says what each service ships, listens on and runs.
set -eu
service=$1 repo=$2 out=$3

app=$out/app bin=$out/bin
mkdir -p "$app/services" "$bin"

# service dir(s) under services/ | shared | top-level dirs | port | uvicorn app (empty: python -m)
case "$service" in
  gateway)       services="gateway";       shared=all; dirs="config prompts"; port=8000
                 uvicorn=services.gateway.src.gateway.main:app ;;
  agent)         services="agent";         shared=all; dirs="config prompts"; port=8002
                 uvicorn=services.agent.src.agent.main:app ;;
  corpus)        services="corpus";        shared=all; dirs="";               port=8001
                 uvicorn=services.corpus.src.corpus.main:app ;;
  ingestion)     services="ingestion";     shared=all; dirs="";               port=8003
                 uvicorn=services.ingestion.src.ingestion.main:app ;;
  # The verifier imports one stdlib-only module of shared/ (the tool specs the agent also reads).
  math_verifier) services="math_verifier"; shared=specs; dirs="";             port=8090
                 uvicorn="" ;;
  *) echo "unknown SERVICE '$service' (gateway, agent, corpus, ingestion or math_verifier)" >&2
     exit 1 ;;
esac

# Top-level files of services/ (package markers) travel with the service directories.
find "$repo/services" -maxdepth 1 -type f -exec cp {} "$app/services/" \;
for s in $services; do cp -R "$repo/services/$s" "$app/services/$s"; done
for d in $dirs; do cp -R "$repo/$d" "$app/$d"; done
case "$shared" in
  all)   cp -R "$repo/shared" "$app/shared" ;;
  specs) mkdir -p "$app/shared/src/shared"
         cp "$repo/shared/src/shared/__init__.py" "$repo/shared/src/shared/math_tool_specs.py" \
            "$app/shared/src/shared/" ;;
esac
find "$app" -name __pycache__ -type d -prune -exec rm -rf {} +

if [ -n "$uvicorn" ]; then
  # exec: uvicorn becomes PID 1's replacement and receives SIGTERM from `docker stop` directly.
  printf '#!/bin/sh\nexec uvicorn %s --host 0.0.0.0 --port %s\n' "$uvicorn" "$port" > "$bin/serve"
  printf '#!/usr/bin/env python\nimport sys, urllib.request\nsys.exit(0 if urllib.request.urlopen("http://localhost:%s/healthz").getcode() == 200 else 1)\n' "$port" > "$bin/healthcheck"
else
  # The verifier reads these itself: bind every interface (other containers reach it) on its port.
  printf '#!/bin/sh\nexport MATH_VERIFIER_HOST=0.0.0.0\nexport MATH_VERIFIER_PORT=${MATH_VERIFIER_PORT:-%s}\nexec python -m services.math_verifier.src.math_verifier.server\n' "$port" > "$bin/serve"
  # FastMCP serves SSE at /sse and has no /healthz: probe that the port accepts connections.
  printf '#!/usr/bin/env python\nimport os, socket\nsocket.create_connection(("localhost", int(os.environ.get("MATH_VERIFIER_PORT", "%s"))), timeout=2).close()\n' "$port" > "$bin/healthcheck"
fi
chmod +x "$bin/serve" "$bin/healthcheck"
