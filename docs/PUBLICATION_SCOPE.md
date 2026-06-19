# Public Repository Scope

This document defines what can and cannot be published in the public GitHub repository.

## Allowed

- Source code in `src/`.
- Infrastructure and automation files in `infra/` and `scripts/`.
- Runtime setup files in `env/`.
- Lightweight, reproducibility-oriented samples in `data/results_sample/`.

## Not allowed

- Any raw dataset or full local dataset mirror.
- Any machine-specific profile with network details (SSID, local IP, host diagnostics).
- Any private credential, token, endpoint, tenant ID, or account-specific metadata.
- Any local cache/build directory and generated temporary artifacts.
- Any manuscript source, figures, or journal submission package files.

## Pre-publish validation

1. Search for secrets and private endpoints.
2. Confirm no large dataset folder is tracked.
3. Confirm all sample files are non-sensitive.
4. Confirm no manuscript files are included in the public package.

## Recommended release practice

- Create a GitHub release tag after initial push.
- Archive the release in Zenodo and add DOI to manuscript Data Availability.
