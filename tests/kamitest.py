# Root for all Kami tests.
# Modifies the path so the kami sources can be imported.

import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'kami'))