In this project, we design and develop a library in which the reviews about the tourism industry are gathered and analyzed from social networks applications (i.e Instagram) to extract relevent knowledge about the context for the decision making process. 

### The project structure overview
There are multiple directories and files in the structure of the project. The directories in this repository are categorized as follows:

#### Data and model:
The datasets and models are stored for the retrieve or save purposes in the following directories:
```
/datasets
/models
```
#### Handlers:
The implementation of the algorithms and processes are kept in the following directories:
```
/concept_extraction
/data_handler
/model_handler
```
#### concept_extraction
The scripts regarding core machine learning algorithms and feature engineering machineries are stored in this directory.
##### data_handler
As it is obvious, it controls the file and data preprocessing operations.
##### model_handler
This directory is responsible for holding the scripts that handle pickling or dumping models for the future usage. 

#### Documentation:
The instructions and manual files of the project are stored here.

#### Misc:
We used this directory to store the test script and temporary files. It is recommend that do not modify or execute the files in this directory.

### Running the tests
In the root directory of the project, there are multiple scripts which call the core algorithms for executing specific operations. The scripts are as follows:
```
test_co_occurrence.py
run_count_vector.py
run_tf_idf.py
run_w2v.py
test_classification.py
```

#### Co-occurrence matrix 

##### Normal execution
In order to test whether the implementation of the co-occurrence matrix works properly
or not, you need to open the ``test_co_occurrence.py`` script under the root directory
of the project. If you normally run the script, you will observe the the co-occurrence matrix in 
the IDE's console. I recommend you to run the script in the debug mode to see the main format of
the matrix. For example, put a breakpoint at line 28 and debuge the application. 

##### Description of the main algorithm

There are two functions in the script namely ``ner_data_document_extraction()`` 
and ``build_co_occurrence_matrix()``. The first function takes a corpra and returns two types of documents as the list type.

1) complete_context:
Considers the whole corpora as the search space.

2) decomposed_context:
Decomposes the main corpra into some documents based on the punctuations in the corpra.
please note that this approach may not be the best solution of the problem.
in order to increase the efficiency of the algorithm, it is better to decompose the context
based on some well-known strategies. 

Once you input the contexts into the ``build_co_occurrence_matrix()``, it returns a list of 
co-occurrence matrices. For instance, if you input the ``decomposed_context``, you will see a list of 
co-occurrence matrix for each document. Otherwise, you will see merely one co-occurrence matrix for the whole document. 
It is worth mentioning that considering the whole document as the context increases the dimensions of the dataset exponentially where for each term in the document, we have a dimension. The solution is
applying some feature extraction algorithms (i.e PCA or SVD) to decompose the matrix and reduce its dimensions. 
 
 ## Programmer
```
Nima Shiri Harzevili (https://github.com/nimashiri)
```
## Acknowledgments

* Thanks god for helping me
