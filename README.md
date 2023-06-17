# Node Resilience Measures, Algorithms and Applications

## Directory structure
* src: contains all the source code 
* data:
    * Contains data set from several different domain.
    * The suffix of the dataset has .txt, .mtx or .edges. To read the graph with other extensions (format), the code may need to modify accordingly. 

## Requirements
<!-- ```bash -->
* python == 3.6 (minimum)
* Install  all of these and possible additional libraries (numpy, networkx, etc.):
    * pip install sympy
    * sudo pip install cycler
    * pip install matplotlib
    * pip install -U scikit-learn scipy matplotlib
    * pip install heapdict

<!-- ```      -->
<!-- ## Dataset:
The dataset can be ,
 ```bash
  
``` -->
## How to run the code?
* **Removal Strength**:
    * python src/dependency_graph_removal.py graph_path SA_type Approach

    * For example, for soc-wiki: 'python src/dependency_graph_removal.py data/soc-wiki-Vote.mtx Subcore Naive' 

    * It will return RS_ID, RS_OD and runtime.

* **Insertion Strength**:
    * python src/dependency_graph_insertion.py  graph_path SA_type Approach

    * For example, for soc-wiki: 'python src/dependency_graph_insertion.py data/soc-wiki-Vote.mtx Traversal Heuristic' 

    * It will return IS_ID, IS_OD and runtime.
* Last two arguments (to run the above code) are described below,
    * **SA_type**: Defines the streaming algorithm type. Options are 1) Subcore 2) Traversal.
    * **Approach**: Defines the approach to calculate the Node Strength. Options are 1) Naive 2) Heuristic.
    * SA_type and Approach are optional arguments. Defaults are Traversal and Heuristic.
    * The runtime performance may differ based on the environment (machine). But, the speedup (Heuristic to Naive) will be similar to paper.

* **Applications**: Figure with comparative performances will be shown after running these codes,
    * Influential Spreader: python src/influential_spreader_application.py graph_path
    * Crtical Edge Removal: python src/edge_removal_application.py graph_path
    * Critical Edge Insertion: python src/edge_insertion_aplication.py graph_path

        Note that, to get the Applications output for large graphs (ca-CondMat, p2p, etc.), it may take longer time to run the code. Because, it will take time to create the dependency graphs. For those graphs, it's better to save the Removal/Insertion strenght first and then use those saved data to get the Applications output.