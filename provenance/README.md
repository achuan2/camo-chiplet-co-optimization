# Provenance

- `source-assets.json`: scientific source/workload hashes; historical Git internals are excluded.
- `dependencies.json`: recorded dependency, package, and image identities.
- `families.json`: independent result directories and record counts.
- `selected-sensitivity-runs.json`: final method/setting/seed selection.
- `raw-release.json`: raw ZIP identity and layout.
- `files.json`: source/public checksums and omitted local-path metadata columns.

Root `MANIFEST.sha256` binds released files except itself. Binary hashes and source-collection hashes identify different objects.

