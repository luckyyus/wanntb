from setuptools import setup, find_packages

setup(name='wann_tb',
      version='2024.11.13',
      description='toolkits based on WF based tight-binding Hamiltonian',
      author='Jie-Xiang Yu',
      author_email='kaelthas.yu@gmail.com',
      license='MIT',
      packages=find_packages(),
      install_requires=['numpy', 'numba'])
