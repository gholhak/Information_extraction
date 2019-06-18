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
The instructions and manual files of the project are stored here (inclduing the readme file).

#### Misc:
We used this directory to store the test script and temporary files. It is recommend that do not modify or execute the files in this directory.

### Running the tests
In the root directory of the project, there are multiple scripts which call the core algorithms for executing specific operations. The scripts are as follows:
```
run_count_vector.py
run_tf_idf.py
run_w2v.py
test_classification.py
```


### And coding style tests

Explain what these tests test and why

```
Give an example
```

## Built With
```
* [Dropwizard](http://www.dropwizard.io/1.0.2/docs/) - The web framework used
* [Maven](https://maven.apache.org/) - Dependency Management
* [ROME](https://rometools.github.io/rome/) - Used to generate RSS Feeds
```
## Contributing
```
Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.
```

## Authors
```
* **Samira Korani** - *Initial work* - [PurpleBooth](https://github.com/PurpleBooth)
```
## Acknowledgments

* Thanks god for helping me
