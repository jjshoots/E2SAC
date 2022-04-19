from setuptools import setup

setup(
    name="DDQN",
    version="1.0.0",
    install_requires=[
        "torch",
    ],
)

setup(
    name="ESDDQN",
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
        "gym[box2d]",
        "opencv-python",
        "pillow",
    ],
)
