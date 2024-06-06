import pkgutil
import importlib

# Automatically import all submodules in the current package
package_name = __name__
__all__ = []

for _, module_name, _ in pkgutil.iter_modules(__path__):
    full_module_name = f"{package_name}.{module_name}"
    module = importlib.import_module(full_module_name)
    
    # Dynamically add the classes from each module to the current namespace
    for attribute_name in dir(module):
        attribute = getattr(module, attribute_name)
        if isinstance(attribute, type):
            globals()[attribute_name] = attribute
            __all__.append(attribute_name)