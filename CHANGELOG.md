# Changelog

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
