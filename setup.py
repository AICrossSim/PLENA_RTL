from setuptools import find_packages, setup

setup(
    name="plena_toolchain",
    version="0.1.0",
    description="Workload generators, cocotb/Verilator runner and RTL tooling for the PLENA accelerator",
    url="https://github.com/AICrossSim/PLENA_RTL",
    license="Apache-2.0",
    python_requires=">=3.12",
    packages=find_packages("tools"),
    package_dir={"": "tools"},
    install_requires=[
        "torch",
        "numpy",
        "cocotb[bus]==1.9.2",
        "bitstring",
        "colorlog",
        "toml",
        "tqdm",
        "pytest",
        "transformers",
        "matplotlib",
    ],
)
