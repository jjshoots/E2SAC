from setuptools import setup

setup(
    name="e2SAC",
    version="1.0.0",
    install_requires=[
        "numpy",
        "wandb",
        "torch",
        "matplotlib",
        "pthflops",
        "gym[box2d]",
        "opencv-python",
        "pillow",
    ],
)

setup(
    name="SAC",
    version="1.0.0",
    install_requires=[
        "wandb",
        "torch",
        "matplotlib",
        "pthflops",
        "gym[box2d]",
        "opencv-python",
        "pillow",
    ],
)
