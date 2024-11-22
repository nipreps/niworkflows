import warnings

msg = 'Niworkflows will be deprecating reporting in favor of a standalone library "nireports".'

warnings.warn(msg, PendingDeprecationWarning, stacklevel=2)
