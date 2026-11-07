# Versions and provenance

## Complete root source

The root contains the common optimizer for the C-TAEA mating-rule study. Recorded source-collection digest:

```text
a89febce6956d13a77722d9cabd8c3e4bedc33b66e1f890acac7dca2b4ee5f15
```

This is a collection digest, not one file's hash. Included `src/moham.cc` SHA-256:

```text
42d6e90175b07e957024711b8734c0b4b277bc5a3f4698b81dff416a9fdd701a
```

`provenance/source-assets.json` binds 256 scientific source/workload payloads. Historical `.git` internals are not distributed. The recorded sensitivity binary is `d96983ff8f243d9efa17c3990d64336fcdaf5fb9ec02621501abed6ed7a63229`.

## Paper result identities

| Family | Recorded binary or source |
|---|---|
| Normal, Temperature-tight, Power-tight, Mobile-3 | Binary `7314342f7043b140785e95fb99b0f3f511e8373283de214e40c24aae26575a07` |
| SLO-tight primary / Datacenter-4 CAMO | Binary `182a6b2e68741adf6de5b720f221dd6d2602a6d7d760ceec12631779a9b09112` |
| Equal-cost / temperature control / Holdout-2 | Binary `141df28f272cc5568f9f022ebf25cdf2ce297a7ef34bcc2931e583823519d64f` |
| Original RQ3 | Source `ab4b8ad29b23168ee2e170a33b636a26fdfc1afcb88489efe5e18c5656942e7d`; binary `05bcc23798fa979e44ef145880d560317aea80d6961207e92bb7a77d65779981` |
| C-TAEA sensitivity | Root collection and binary above |

Original SLO-tight/Datacenter-4 comparison-method runs retain the first binary; final CAMO runs use the second. `snapshots/` contains retained fixed-budget/RQ3 excerpts, not alternative complete build trees.

Final sensitivity inputs select 649 original runs and 11 accepted replacements. `provenance/selected-sensitivity-runs.json` records method/setting/seed selection, and the raw Release index binds every CSV.

## Dependencies and caches

Timeloop: `34ba2aa84f47c319a5958af97d776683ea45d453`. Its preserved source payload, including compatibility changes, is included. Accelergy/plugin revisions are pinned in the recorded Docker recipe. `provenance/dependencies.json` records historical image and package identities; a rebuilt image receives its own identity.

Recorded read-only mapping-cache tree digests:

- Edge-3: `2ad5200b0dd5664826d52fe55bc9758a08c06f53fd9d46d27054a8ab0354c580`.
- Holdout-2: `3a38c8f1be62bb748d894023c1bdc5177ab7b63801bae5e2a55a25b447322e27`.

Generated cache trees are not ordinary Git files in this package. Portable examples rebuild them from included graphs. Saved metrics and selected raw CSVs support result reconstruction independently.

## Optional scalar-CV ordering correction

The archived scalar-CV control's pairwise tolerance/lexicographic rule can be nontransitive. `patches/scalar_cv_strict_order.patch` supplies the separately named strict-order correction and an isolated C++ regression source. Root code and saved results retain their recorded version. Applying the patch creates a different control version and does not rewrite preserved results.

