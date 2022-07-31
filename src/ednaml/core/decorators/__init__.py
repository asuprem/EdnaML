import inspect, os

import ednaml


REGISTERED_EDNA_COMPONENTS = {}


def register(func, functype):
    try:
        fname = inspect.getfile(func)
    except TypeError:
        fname = os.path.abspath(func.__module__)
    print("Registering a %s: %s, from file: %s"%(functype, str(func), fname))
    if fname not in REGISTERED_EDNA_COMPONENTS:
        REGISTERED_EDNA_COMPONENTS[fname] = {}
    # TODO also add in some type of lookup for the file basename...?
    if functype is "model_plugin":
        if "model_plugin" not in REGISTERED_EDNA_COMPONENTS[fname]:
            REGISTERED_EDNA_COMPONENTS[fname]["model_plugin"] = []
        REGISTERED_EDNA_COMPONENTS[fname]["model_plugin"].append(func)
    else:
        REGISTERED_EDNA_COMPONENTS[fname][functype] = func


def register_crawler(func):
    register(func, "crawler")
    return func

def register_model(func):
    register(func, "model")
    return func

def register_generator(func):
    register(func, "generator")
    return func

def register_storage(func):
    register(func, "storage")
    return func

def register_trainer(func):
    register(func, "trainer")
    return func

def register_deployment(func):
    register(func, "deployment")
    return func


def register_model_plugin(func):
    register(func, "model_plugin")
    return func

def register_model_pre_plugin(func):
    pass
