import io
import re
from setuptools import setup, find_packages

from eeg_to_audio import __version__

def read(file_path):
    with io.open(file_path, 'r', encoding='utf-8') as f:
        return f.read()


readme = read('README.rst')
# вычищаем локальные версии из файла requirements (согласно PEP440)
requirements = '\n'.join(
    re.findall(r'^([^\s^+]+).*$',
               read('requirements.txt'),
               flags=re.MULTILINE))


setup(
    # metadata
    name='eeg_to_audio',
    version=__version__,
    license='MIT',
    author='Muhammadsharif Nabiev',
    author_email="nabiev.mf@phystech.edu",
    description='eeg_to_audio, python package',
    long_description=readme,
    url='https://github.com/intsystems/2024-Project-154',

    # options
    packages=find_packages(),
    install_requires=requirements,
)
