## Directory Structure
This folder stores the files common to pytorch and other frameworks, e.g., JAX. The build process builds a shared library for the common files and links it when building the pytorch extension. During python runtime, the shared library in this common folder will be loaded by ctypes via RTLD_GLOBAL mode.

## Reference 
[TransformerEngine/transformer_engine/common - Github](https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/common)

[ctypes loading a c shared library that has dependencies - StackOverflow](https://stackoverflow.com/a/30845750)