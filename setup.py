from setuptools import setup, find_packages

setup(
    name="pixsi",
    version="0.1",
    packages=find_packages(),
    install_requires=["click","numpy", "matplotlib", "scipy", "torch>=2.6.0", "sortedcontainers", "h5py", "tqdm"],
    description="Signal processing tools for LarPix ND DUNE",
    entry_points = dict(
        console_scripts = [
            'pixsi = pixsi.__main__:main',
        ]
    ),
)
