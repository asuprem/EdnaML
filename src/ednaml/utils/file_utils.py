import mmap
from types import FunctionType



class IterableFile:

    def __init__(self, file_path, chunk_size = 50000, line_callback: FunctionType = None):
        # TODO use chunk-size to preload some chunk into memory for ammortized overhead?
        self.row_count = 0
        for _ in (row for row in open(file_path, "r")):
            self.row_count+=1
        self.file_path = file_path
        self.line_callback = line_callback

    def __len__(self):
        return self.row_count

    def __iter__(self):
        return (self.line_callback(row) for row in open(self.file_path, "r"))


"""
self.file_obj = open(file_path, "r+b")

        try:    # UNIX vs Windows
            s = mmap.PROT_READ
        except AttributeError:
            s = mmap.ACCESS_READ

        self.mmap = mmap.mmap(self.file_obj.fileno(), 0, prot=s)
        self.mmap_iter = iter(self.mmap.readline, b"")
        
"""