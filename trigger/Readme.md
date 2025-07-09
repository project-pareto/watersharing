PREREQUESITE:
Microsoft Visual C++ 14.0

TO RUN LOCALLY FOR FIRST TIME:

1. cd into `watersharing-py/`

2. In terminal enter: `conda create -n pareto-env python=3.9 pip --yes`
   followed by `conda activate pareto-env`

3. Next install the requirements by entering `pip install -r requirements-dev.txt`

4. Then enter this into the terminal as well `idaes get-extensions --verbose`

5. Next, `conda install --channel conda-forge pyscipopt`

6. Make sure you have numpy installed: `conda install numpy=1.25.1`

7. Make sure you have glpk installed: `conda install -c conda-forge glpk`

8. In terminal, to run the matching scripts locally, type `python codeTrigger.py --mode local` and then move a correctly formatted json file into ./export_local

9. After running, a few files should be created with matches and distances in JSON format within the import_local folder

**If you run into errors regarding SCIP on MacOS, try using [Homebrew](https://brew.sh/) to [install scip](https://formulae.brew.sh/formula/scip)**

**If you would like to keep the trigger actively searching for new matches(similar to the intended use), run it via `python codeTrigger.py &` to run the script in the background. You may kill this process by typing `jobs -l` finding the identifying number for the process and then typing `kill <id>` You can test the json file generation by either copying a json file into the ./export_local folder or setting up a local MAMP environment using the watersharing wordpress repo and redirecting the local path(config.yaml) to the export folder in /io/export. Upon filling out one of the forms a json file will be sent to the export folder and matches should be created as intended**

ON THE SERVER:

1. Run the trigger script using the command `nohup python codeTrigger.py &  --mode production`. All the 'nohup' prefix is doing is keeping the script running after you close the ssh window

2. Check the trigger is running by typing `ps -ef | grep “python codeTrigger.py"` instead of what is mentioned above. Killing the trigger process can be done similarly using the kill command and the id you find using the aforementioned command

ADDITIONAL NOTES:
**You may change the path which the code is using to search for new json files by modifying the entries within config.yaml. The local_vars are meant to test with files in the local directory and the production_vars are meant to point at a wordpress directory**
