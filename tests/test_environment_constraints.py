"""
Test environment constraints for Immune2Vec functionality.
"""

import sys

import pytest


class TestEnvironmentConstraints:
    """Test that the environment meets Immune2Vec requirements."""

    def test_python_version_compatibility(self):
        """Test that Python version is compatible with Immune2Vec."""
        version = sys.version_info

        # Python 3.8+ is required for optimal gensim 3.8.3 compatibility
        assert version.major == 3, f"Python 3.x required, got {version.major}.{version.minor}"
        assert version.minor >= 8, f"Python 3.8+ required for Immune2Vec, got {version.major}.{version.minor}"

        print(f"✅ Python version {version.major}.{version.minor}.{version.micro} is compatible")

    def test_gensim_version_exact(self):
        """Test that gensim version is exactly 3.8.3 as required by Prerequisites."""
        try:
            import gensim
        except ImportError:
            pytest.skip("gensim not installed")

        expected_version = "3.8.3"
        actual_version = gensim.__version__

        assert actual_version == expected_version, (
            f"Immune2Vec Prerequisites require gensim=={expected_version}, "
            f"but found {actual_version}. "
            f"Install with: python3 -m pip install gensim=={expected_version}"
        )

        print(f"✅ gensim version {actual_version} matches Prerequisites")

    def test_ray_availability(self):
        """Test that ray package is available as required by Prerequisites."""
        try:
            import ray  # noqa: F401
        except ImportError:
            pytest.fail("ray package is required by Immune2Vec Prerequisites. " "Install with: pip3 install ray")

        print("✅ ray package is available")

    def test_immune2vec_prerequisites_complete(self):
        """Test that all Immune2Vec Prerequisites are met."""
        # Test Python version
        version = sys.version_info
        assert (
            version.major == 3 and version.minor >= 8
        ), f"Prerequisites require python3.*, got {version.major}.{version.minor}"

        # Test gensim version
        try:
            import gensim

            assert gensim.__version__ == "3.8.3", f"Prerequisites require gensim==3.8.3, got {gensim.__version__}"
        except ImportError:
            pytest.fail("Prerequisites require gensim==3.8.3 to be installed")

        # Test ray availability
        try:
            import ray  # noqa: F401
        except ImportError:
            pytest.fail("Prerequisites require ray package to be installed")

        print("✅ All Immune2Vec Prerequisites are satisfied")
        print(f"   - Python: {version.major}.{version.minor}.{version.micro}")
        print(f"   - gensim: {gensim.__version__}")
        print("   - ray: available")
