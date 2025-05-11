import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


class PackageImportTests(unittest.TestCase):
    def test_import_package_and_version(self):
        import lcdm_sim

        self.assertTrue(hasattr(lcdm_sim, "__version__"))
        self.assertIsInstance(lcdm_sim.__version__, str)
        self.assertTrue(lcdm_sim.__version__)


if __name__ == "__main__":
    unittest.main()
