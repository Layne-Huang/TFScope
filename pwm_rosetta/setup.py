import setuptools as tools

tools.setup(
    name="pwm_rosetta",
    version="0.1.0",
    description="Hybrid AF3 + PyRosetta PWM generation for transcription factor binding sites",
    author="Lei Huang",
    packages=[
        'pwm_hybrid',
        'pwm_hybrid.af3',
        'pwm_hybrid.rosetta',
        'pwm_hybrid.pwm',
    ],
    package_dir={
        'pwm_hybrid': './pwm_hybrid',
    },
    package_data={
        # Bundle Rosetta weight/flag files and the xml_loader module
        'pwm_hybrid.rosetta': [
            '_data/flags_and_weights/RM8B_flags',
            '_data/flags_and_weights/RM8B_torsional.wts',
            '_data/flags_and_weights/no_ref.rosettacon2018.beta_nov16_constrained.txt',
            '_xml_loader.py',
        ],
    },
    install_requires=[
        'numpy',
        'pandas',
        'matplotlib',
        'scipy',
        'biopython',
        'gemmi',
        'tqdm',
    ],
    extras_require={
        'viz': ['logomaker'],
    },
    entry_points={
        'console_scripts': [
            'pwm-hybrid=pwm_hybrid.cli:main',
        ],
    },
    python_requires='>=3.9',
)
