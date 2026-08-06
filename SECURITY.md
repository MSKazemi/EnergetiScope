# Security Policy

## Supported versions

EnergetiScope is pre-1.0 research software. Only the `main` branch is supported — fixes land
there, and there are no backports to older tags.

| Version | Supported |
|---|---|
| `main` | ✅ |
| tagged releases | ❌ (upgrade to `main`) |

## Reporting a vulnerability

**Please do not open a public issue for a security problem.**

Use GitHub's [private vulnerability reporting](https://github.com/MSKazemi/EnergetiScope/security/advisories/new)
on this repository. If that is unavailable to you, email
**mohsen.seyedkazemi@gmail.com** with `EnergetiScope security` in the subject.

What to expect:

- **Acknowledgement within about a week.** This is a single-maintainer project; see
  [`CONTRIBUTING.md`](CONTRIBUTING.md) for the honest response-time picture.
- An assessment of whether it is exploitable and in what deployment.
- Credit in the fix commit and release notes, unless you prefer to stay anonymous.

## Threat model — what actually matters here

EnergetiScope is not an authentication or data-protection system, so the realistic risks are
specific. These are in scope and worth reporting:

- **The prediction service parses untrusted input.** `/predict/from-yaml` and
  `/infer/from-yaml` accept arbitrary YAML. Parser abuse, resource exhaustion, or anything
  that escapes the parser is in scope.
- **Model artifacts are `joblib`/pickle files.** Loading a `.joblib` from an untrusted source
  executes arbitrary code — this is inherent to the format, not a bug in this project.
  `ENCODER_PATH` and `MODEL_PATH` must only ever point at artifacts you trust. Report it if
  the service can be made to load an artifact from an attacker-controlled path.
- **The collector holds cluster read credentials.** `k8s_collect.py` runs with a ServiceAccount
  that can read workload specs across namespaces. Anything that leaks those specs, escalates
  beyond read, or widens the RBAC in `k8s/jobs/01-rbac.yaml` is in scope.
- **Collected workload specs are sensitive.** Image names, labels, and annotations can reveal
  internal infrastructure. Report any path that writes them somewhere unintended.

Out of scope: the service ships **no authentication** and is designed to run inside a cluster
behind your own network policy. "The API is unauthenticated" is a documented property, not a
vulnerability — but do tell us if the docs fail to make that clear enough.

## Deployment note

Do not expose the prediction service directly to the public internet. Put it behind your
cluster's network policy and ingress authentication.
