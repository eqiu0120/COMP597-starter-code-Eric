# COMP597-starter-code
This repository contains starter code for COMP597: Responsible AI - Energy Efficiency analysis using CodeCarbon. 
TODO: add more description on for course description, project description and instructions on the project.
### Course Description
TODO: course description.

### Project Description
TODO: project description.

### Instructions
TODO: instructions for the project. eg:
1. Set up your environment using the provided instructions below under [Environment Setup](#environment-setup).
2. Familiarize yourself with the CodeCarbon library and its usage. Resources can be found in the [CodeCarbon Resources](#codecarbon-resources) section.
3. Implement your language/vision/other model and run experiements to collect data.
4. Document your process and findings in a report.

---

TODO: add a section for repository structure. with a tree and description of important folders and files.

Structure: #TODO
```
COMP597-starter-code
.
├── README.md                               # instructions and description of the project
├── requirements.txt                        # list of required packages to install
├── env.sh                                  # script to setup the conda environment variables
├── .gitignore                              # gitignore file to exclude unnecessary files
├── ...
├── energy_efficiency
│   ├── src
│   │   ├── config
│   │   │   └── config.py                   # configurations file with user given arguments
│   │   ├── models
│   │   │   ├── gpt2
│   │   │   │   ├── __init__.py
│   │   │   │   └── gpt2.py                 # gpt2 model simple trainer example
│   │   ├── trainer
│   │   │   ├── stats
│   │   │   │   ├── codecarbon.py           # trainer stats to collect codecarbon information with losses
│   │   │   │   └── ...
│   │   │   ├── base.py                     # abstract methods 
│   │   │   ├── simple.py                   # simple trainer 
│   │   │   └── ...
│   ├── launch.py
│   ├── start-gpt2.sh                       # script to easily start gpt2 
│   └── ...
└── ...
```

#### environment setup

We will use a Conda envrionment to install the required dependencies. The steps below will walk you through the steps. A setup script `env_setup.sh` is also provided and will execute all the steps below given as input the path `SOME_PATH` as described in step one below.

1. **Setting up storage** <br> Your home directory on the McGill server is part of a network file system where users get limited amounts of storage. You can check your storage usage and how much you are allowed to use using the command `quota`. Python packages, pip's cache, Conda's cache and datasets can use quite a bit of storage, so we need to ensure they are stored outside your directory to avoid any issues with disk quotas. Say you have your own directory, stored in `SOME_PATH`, on a server that is not part of the network file system (hence not affected by disk quotas). Use `export BASE_STORAGE_PATH=SOME_PATH` where you replace `SOME_PATH` with the actual path. The steps to go around the disk quota are as follows:
    1. We can make a cache directory using `mkdir ${BASE_STORAGE_PATH}/cache`. 
    2. For pip's cache, we can redirect it to that directory using `export PIP_CACHE_DIR=${BASE_STORAGE_PATH}/cache/pip`. 
    3. For Hugging Face datasets, we can use `export HF_HOME=${BASE_STORAGE_PATH}/cache/huggingface`. While this variable is not strictly needed for the environment set up, it is needed when using the Hugging Face datasets module.
2. **Initializing Conda** <br> If you have never used Conda with this user, you need to initialize Conda with `conda init bash`. This modifies the `~/.bashrc` file. Unfortunately, the `~/.bashrc` file is not always executed at login, depending on the server configurations. For that reason, it is recommended to run `. ~/.bashrc` before running any Conda commands. 
3. **Creating the project environment** <br> First, let's make sure to create the directory to store the environment using `mkdir -p ${BASE_STORAGE_PATH}/conda/envs`. You can now simply run `conda create --prefix ${BASE_STORAGE_PATH}/conda/envs/COMP597-project python=3.14` to create the environment. 
4. **Activating the environment** <br> You can use your environment by activating it with `conda activate ${BASE_STORAGE_PATH}/conda/envs/COMP597-project`. 
5. **Installing dependencies** <br> The dependencies are provided as a requirements file. You can install them using `pip install -r energy_efficiency/requirements.txt`.
6. **Using the environment** <br> For any future use of the environment, you can create a script, let's name it `local_env.sh`, which will contain the configuration to set up the environment. You can then execute the script with `. local_env.sh` to set up activate your environment. The script would look like this (where you need to replace `SOME_PATH`.:
    ```
    #!/bin/bash
    
    . ~/.bashrc
    conda activate SOME_PATH/conda/envs/COMP597-project
    export PIP_CACHE_DIR=SOME_PATH/cache/pip
    export HF_HOME=SOME_PATH/cache/huggingface
    ```
7. **Quitting** <br> If you want to quit the environment, or reset your sheel to before you activate the environment, simply run `conda deactivate`.

TODO: add section for resources
### CodeCarbon Resources
- [olivier-tutorial-code-carbon](https://colab.research.google.com/drive/1eBLk-Fne8YCzuwVLiyLU8w0wNsrfh3xq)
- [laura-documentation](https://docs.google.com/document/d/1GSxPYXRVjkb1eSwnZZBsSQMVICRLdM_pM-iEVeivAIE/edit)
- [laura documentation](https://docs.google.com/document/d/1Ihfniv1CaWz79tO4IcXx3JG7pAZDIGWigAMDKiVTNDc/edit)

TODO: add section for how to run experiments, how to edit files to add a new model etc.

---
