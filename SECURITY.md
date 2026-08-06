# Security policy

## Reporting

If you find a vulnerability in this repository (secrets leak, XSS in the static UI,
unsafe deserialization), open a private report via GitHub Security Advisories or
email the maintainer listed on the repo.

## Scope

- Static GitHub Pages UI (`docs/`) — XSS / content injection
- Model artifacts (`.joblib`) — treat as untrusted binary; SHA256 sidecars are checked on load
- API keys — never commit `.env`; OpenDota key is optional

## Out of scope

- OpenDota / third-party API availability
- Prediction accuracy / gambling outcomes

## Hardening notes

- Prefer `joblib` artifacts with SHA256 verification over pickle scripts (legacy pickle lives under `legacy/`)
- UI escapes user/team strings before `innerHTML`
- CI does not inject secrets into Pages builds
