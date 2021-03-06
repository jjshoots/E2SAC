from setuptools import setup

setup(
    name="e2SAC",
    version="1.0.0",
    install_requires=[
        "torch",
    ],
)

setup(
    name="SAC",
    version="1.0.0",
    install_requires=[
        "torch",
    ],
)

setup(
    name="utils",
    version="1.0.0",
    install_requires=[
        "wandb",
        "torch",
        "matplotlib",
        "pthflops",
        "opencv-python",
        "pillow",
    ],
)
