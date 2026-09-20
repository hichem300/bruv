# Security policy

## Reporting a vulnerability

Please report security issues privately. Do not open a public issue. Email the
maintainers with a description and, if possible, a minimal reproduction. You
will receive an acknowledgement within a reasonable timeframe.

## Scope

bruv reads credentials only from the `TYPESAFE_API_KEY` environment variable or
a protected credential file. It never accepts API keys as CLI flags. Report
issues that expose credentials, bypass no-paid-call guarantees, or break
calibration disclosure.

## Disclosure

We coordinate disclosure and credit reporters once a fix is available.
