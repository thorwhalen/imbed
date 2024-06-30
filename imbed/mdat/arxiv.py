"""Access to ArXiv data.

Project moved to xv: https://pypi.org/project/xv/

"""

from contextlib import suppress


with suppress(ImportError):
    from xv.data_access import *  # pip install xv
