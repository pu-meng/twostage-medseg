from setuptools import setup, find_packages

setup(
    name="twostage_medseg",
    packages=["twostage_medseg"] + ["twostage_medseg." + p for p in find_packages()],
    package_dir={"twostage_medseg": "."},
)
