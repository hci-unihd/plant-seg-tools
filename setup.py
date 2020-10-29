from setuptools import setup, find_packages

exec(open('plantsegtools/__version__.py').read())
setup(
    name='plantsegtools',
    version=__version__,
    packages=find_packages(exclude=["tests", "evaluation"]),
    include_package_data=True,
    description='Suite of Python tools for plant-seg https://github.com/hci-unihd/plant-seg',
    author='Lorenzo Cerrone, Adrian Wolny',
    url='https://github.com/hci-unihd/plant-seg-tools',
    author_email='lorenzo.cerrone@iwr.uni-heidelberg.de',
)