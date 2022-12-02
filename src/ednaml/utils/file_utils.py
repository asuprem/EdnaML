from types import FunctionType


class IterableFile:
    def __init__(self, file_path, chunk_size=50000, line_callback: FunctionType = None):
        # TODO use chunk-size to preload some chunk into memory for ammortized overhead?
        self.row_count = self._bufcount(file_path)
        self.file_path = file_path
        self.line_callback = line_callback
        self.file_obj = open(self.file_path, "r")

    def __len__(self):
        return self.row_count

    def __iter__(self):
        return (self.line_callback(row) for row in self.file_obj)


    def _bufcount(self, filename):
        f = open(filename)                  
        lines = 0
        buf_size = 1048576 # 1024 * 1024
        read_f = f.read # loop optimization

        buf = read_f(buf_size)
        while buf:
            lines += buf.count('\n')
            buf = read_f(buf_size)

        return lines

    def close(self):
        self.file_obj.close()

"""
self.file_obj = open(file_path, "r+b")

        try:    # UNIX vs Windows
            s = mmap.PROT_READ
        except AttributeError:
            s = mmap.ACCESS_READ

        self.mmap = mmap.mmap(self.file_obj.fileno(), 0, prot=s)
        self.mmap_iter = iter(self.mmap.readline, b"")
        
"""
