# Changelog

## Unreleased

### Security

- Upgraded `transformers` dependency from `<4.52` to `>=5.3.0` to address a critical
  remote code execution vulnerability (Dependabot alert #7, CVE-2026-4372) affecting
  all versions of the HuggingFace transformers library prior to 5.3.0.

### Fixed

- Added a compatibility patch for AntiBERTy (`antiberty` ≤ 0.1.3, last released
  July 2023) to work with `transformers` ≥ 4.52.
  - `transformers` ≥ 4.52: replaces the deprecated `init_weights()` call with
    `post_init()` in `AntiBERTy.__init__`, fixing
    `AttributeError: 'AntiBERTy' object has no attribute 'all_tied_weights_keys'`
    (see https://github.com/jeffreyruffolo/AntiBERTy/issues/4).
  - `transformers` ≥ 4.36: forces `attn_implementation="eager"` when loading the
    AntiBERTy model, preserving numerical consistency with the original embeddings
    (SDPA, the new default, produces different values).

## 2.1.2 - Green Weka hotfix II

### Fixed

- Improved TCRT5 and AbLang parallelization.
- Minor doc fixes.
- Added multithread processing for translate-igblast.

## 2.1.1 - Green Weka hotfix

### Fixed

- Fixed automatic locus determination.
- Fixed example command in documentation.

## 2.1 - Green Weka

### Added

- Added fine-tuning tutorial

### Fixed

- Make locus uppercase to avoid missmatches.
- Fixed bug on translation when `sequence_vdj` column was empty.
- Fix compatibility for Python 3.14 (does not support AbLang).

## 2.0 - Brown Kiwi

### Added

- Allow calling Python API as well as CLI.
- Allow choosing sequence column for translation.
- Added PROTT5, TCRT5 and TCRbert embeddings.
- Added Ablang embeddings.
- Added AnnData export
- Added possibility to store residue-level embeddings for all models that support them.

### Fixed

- Refactored code to allow extending to TCR embeddings.
- Used lazy imports to decrease the response time to load help message.

## 1.1 - Blue Tui

### Added

- Added explicit command for BALM-paired embedding
- Added Tutorial
- Added better docs on included embeddings

### Fixed

- When multiple heavy or light chains are present, pick the one with highest duplicate count, not consensus count.
- Consider IGH, TRB and TRD as heavy chains.

## 1.0 - Green kea

- First release on PyPI of amulety
