# Environment Setup

This project uses an Anaconda environment that is created using a .yml file.
To recreate the environment, it is nessary to have the Anaconda distribution, which can be downloaded here: https://www.anaconda.com/download

Once it has been installed, follow these steps:

- Step 1: Open your terminal (or Anaconda Prompt).
- Step 2: Navigate to the folder containing <environment.yml> file.
- Step 3: Run the following command to create the environment: $conda env create -f environment.yml
- Step 4: Activate the newly created environment : $conda activate intro_ai_env

After following these steps, the activated environment should have all the required dependencies to run the software.


# Jupyter Notebook

Once the environment is activated, start Jupyter Notebook from the terminal (running $jupyter notebook) or using the anaconda distribution.

This ensures that your Jupyter session uses the correct environment and its dependencies.

# Running the Code
Once your environment is set up and the Jupyter Notebook is launched, navigate to the notebook file in your browser and run the cells to execute the project code.