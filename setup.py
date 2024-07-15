from setuptools import setup, find_packages

setup(
    name='phy_cut_paste',
    version='0.1.0',
    description='Cut-And-Paste Data Augmentation for Multiple Annotations',
    author='Jack Mead',
    author_email='your@email.com',
    url='https://github.com/ainascan/phy_cut_paste',
    packages=find_packages(),
    install_requires=[
        "pymunk",
        "numpy",
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
)