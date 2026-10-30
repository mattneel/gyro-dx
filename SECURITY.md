# Security policy

Report vulnerabilities privately through GitHub's
[private vulnerability reporting](https://github.com/mattneel/gyro-dx/security/advisories/new).
Do not open a public issue.

The GGUF reader parses untrusted files.
It bounds metadata reads and checks arithmetic for overflow.
The validation also covers tensor ranges, overlap, and alignment.
Reports about those checks are in scope.
So are reports about the native bridge's range validation and the HTTP server.

The HTTP server binds to `127.0.0.1` by default.
It has no authentication.
Do not expose it on an untrusted network.
