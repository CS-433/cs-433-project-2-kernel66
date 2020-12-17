# Machine Learning, Project Ebola

### Setup:

To run our notebooks, the installation of a number of libraries is required. 
First, basic libraries for scientific computing and data manipulation:
  - numpy
  - pandas
  - sys
  
Then, libraries for data visualization:
  - matplotlib
  - graphviz
  - seaborn

Finally, libraries for statistics and machine learning: 
  - statsmodels
  - sklearn
  - yellowbrick
  - eli5
 
An additional library to calculate the carbon footprint:
  - cumulator 

With the package manager [pip](https://pip.pypa.io/en/stable/) the installation of the latest version of one of the libraries is done by this way:
```sh
$ pip install SomePackage
```
Some third-party distributions also provide versions of most of these libraries, integrated with their package-management systems. 
For example, to install a package with anaconda:
```sh
$ conda install SomePackage
```
For further information go to  [docs.python.org](https://docs.python.org/fr/3.6/installing/index.html). 

### Installation:

The git repository contains four folders:
- `study_EIXUZQ`
- `study_EGOYQN`
- `Python_files`
- `Data`

For each of the two studies, the distribution of the files is as follows. 
    
    ├── ...
        ├── study_EIXUZQ                  
        │   └── study_EIXUZQ.ipynb
        ├── study_EGOYQN  
        │   ├── study_EGOYQN_export_dataframe.ipynb
        │   ├── study_EGOYQN_linelist_diagnosis.ipynb
        │   ├── study_EGOYQN_linelist_prognosis.ipynb
        │   └── study_EGOYQN_clinical.ipynb
        ├── Python_files
        │   ├── EGOYQN_linelist_diagnosis_results.py
        │   ├── EGOYQN_clinical_prognosis_results.py
        │   └── proj2_HELPERS_.py
        ├── Data
        └── README.md
    
Running the notebooks in each study will recreate our predictions, however it needs access to the data from the EPFL/MLO-IDDO collaborative project. Please contact Dr Mary-Anne Hartley or Ridha Chahed in order to get access to the dataset (mary-anne.hartley@epfl.ch and mohamed.chahed@epfl.ch). Then please insert the following .csv data files into the `Data` folder.

| Data files |
| ------ |
| Clinic_test.csv |
| Clinic_train.csv |
| EGOYQN_Gueckedou_clinFU_cleaned_pos_and_neg_2019_November.csv |
| EGOYQN_Gueckedou_linelist_cleaned_pos_and_neg_2019_November_ANSI.csv |
| EIXUZQ_LIB_FOYA.xls |
| Help_frame.csv |
| linelist_out.csv |
| Linelist_test.csv |
| Linelist_train.csv |

### Most important notebooks of the project 

The most important notebooks, i.e. those containing the vast majority of our work, are the following:

`study_EIXUZQ` → `study_EIXUZQ.ipynb`

`study_EGOYQN` → `study_EGOYQN_linelist_diagnosis.ipynb`, `study_EGOYQN_linelist_prognosis.ipynb`, `study_EGOYQN_clinical.ipynb`


### `Python_files` folder 

In the `Python_files` folder you will find the `project2Helpers.py` which holds functions used throughout the project. You will also find `EGOYQN_linelist_diagnosis_results.py` which enables you to run our best model for predicting Ebola and, respectively, `EGOYQN_clinical_prognosis_results.py` which will run our best model for predicting the outcome of the patients. 

### Authors list

Authors: Lavinia Schlyter, Cédric Roy, Jean Naftalski 
Supervisor and help: Ridha Chahed

