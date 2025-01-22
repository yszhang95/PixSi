from setuptools import setup, find_packages

setup(
    name="pixsi",
    version="0.1",
    packages=find_packages(),
    install_requires=["click","numpy", "matplotlib", "scipy"],
    description="Signal processing tools for LarPix ND DUNE",
    entry_points = dict(
        console_scripts = [
            'pixsi = pixsi.__main__:main',
        ]
    ),
)
