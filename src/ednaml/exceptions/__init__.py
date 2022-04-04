

#https://github.com/python/cpython/blob/aa8ea3a6be22c92e774df90c6a6ee697915ca8ec/Lib/pydoc.py#L288
class ErrorDuringImport(Exception):
    """Errors that occurred while trying to import something to document it."""
    def __init__(self, filename, exc_info):
        self.filename = filename
        self.exc, self.value, self.tb = exc_info

    def __str__(self):
        exc = self.exc.__name__
        return 'problem in %s - %s: %s' % (self.filename, exc, self.value)