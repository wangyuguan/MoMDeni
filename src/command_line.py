"""Back-compat shim: ``python -m command_line`` now lives in kam.command_line.

Kept so scripts generated before the package restructure keep working; new
code should use the installed ``kam`` console command.
"""

import sys

from kam.command_line import main

if __name__ == "__main__":
    sys.exit(main())
