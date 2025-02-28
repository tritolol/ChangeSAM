from setuptools import find_packages, setup

setup(
    name="changesam",
    version="1.0",
    install_requires=[
        "tqdm",
        "scikit-learn>=1.6.1",
        "pillow>=11.1.0",
        "requests>=2.32.3",
        "segment_anything @ git+https://github.com/facebookresearch/segment-anything.git"
    ],
    packages=find_packages(),
    scripts=["tools/changesam_train.py"
             "tools/changesam_test.py",
             "tools/changesam_precompute_embeddings.py"]
)