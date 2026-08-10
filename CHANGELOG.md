# Changelog

## 2.1.3 - Green Kakapo

### Fixed

- Create output directory in the translate-igblast command if it doesn't exist.
- Allow different prefixes and organism names for igblast germline reference files.
- Pinned version of transformers to avoid incompatibility with Antiberty.

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
