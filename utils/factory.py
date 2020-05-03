import importlib


def get_class(module_name: str, class_name: str):
    print(f"Importing class {class_name} from module {module_name}...")
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)
    return cls
