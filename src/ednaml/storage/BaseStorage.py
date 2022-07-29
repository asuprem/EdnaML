class BaseStorage:
    storage_type: str
    storage_url: str
    def __init__(self, type, url, **kwargs):
        self.storage_type = type
        self.storage_url = url
        self.build_params(**kwargs)
        
    def build_params(self, **kwargs):
        pass

    def read(self):
        print("Base read call")

    def write(self, data):
        print("Base write call",data)

    def append(self,data):
        print("Append call",data)

    def copy(self,src):
        print("Copy call ",src)
