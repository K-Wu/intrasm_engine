
## Development
This repository uses setup.py to build the package. To develop, install the package in editable mode:
```
pip install -e .
```

### Reference
[TransformerEngine/transformer_engine - Github](https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/)

[Python setup.py develop vs install - StackOverflow](https://stackoverflow.com/a/19048754)

## Contributing
### Pybind Overloading
[Binding Disambiguation - pybind11 Documentation](https://pybind11.readthedocs.io/en/stable/classes.html#:~:text=We%20can%20disambiguate%20by%20casting%20them%20to%20function%20pointers)

[Adding Lambda Function as a Class Method - pybind11 Documentation](https://pybind11.readthedocs.io/en/stable/classes.html#:~:text=Unfortunately%2C%20there%20is%20no%20suitable%20functionality%20in%20the%20Pet%20data%20structure%2C%20and%20it%20would%20be%20nice%20if%20we%20did%20not%20have%20to%20change%20it.)

### Pybind Resource Management
[Smart Pointers](https://pybind11.readthedocs.io/en/stable/advanced/smart_ptrs.html)

## Directory Structure
This repository follows the directory structure of [TransformerEngine - Github](github.com/NVIDIA/TransformerEngine/).

## Code Health Badges
[![CodeFactor](https://www.codefactor.io/repository/github/k-wu/intrasm_engine/badge?s=749489c3b14056d2ece1446c9f6f3e55572069b3)](https://www.codefactor.io/repository/github/k-wu/intrasm_engine)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/efbb131ba609458c8a586ea63c2534e2)](https://app.codacy.com?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade)
[![DeepSource](https://app.deepsource.com/gh/K-Wu/intrasm_engine.svg/?label=active+issues&show_trend=true&token=OE3XZsUS8QPEMWILgPiJbtGG)](https://app.deepsource.com/gh/K-Wu/intrasm_engine/)

## Auto-tuning
### Microbenchmark
We use the [Accel-Sim microbenchmark suites](https://github.com/accel-sim/gpu-app-collection/blob/release/src/cuda/GPU_Microbenchmark/), which is based on ["Dissecting the NVIDIA Volta GPU Architecture via Microbenchmarking"](https://arxiv.org/pdf/1804.06826.pdf)