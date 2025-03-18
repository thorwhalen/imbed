"""Data access and preparation modules

Forwards to the imbed_data_prep package.
"""

# Forwarding modules to imbed_data_prep

from lkj import register_namespace_forwarding

register_namespace_forwarding("imbed.mdat", "imbed_data_prep")
