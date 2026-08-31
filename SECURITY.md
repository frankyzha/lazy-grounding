# Security policy

## Reporting a vulnerability

Please report security issues privately through GitHub's security-advisory
interface. Do not include API keys, private benchmark items, or complete agent
traces in a public issue.

## Credential handling

The software reads credentials only from environment variables or a local
`.env` file. Credential files, raw traces, and experiment outputs are ignored by
Git. Run `gitleaks git .` before every public release.

## Responsible use

Do not publish benchmark-targeted nearby-evidence pages to the public web. The
controlled retrieval environment exists to evaluate exposure without polluting
search indexes or future benchmark evaluations.
