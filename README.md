# LSH for 3d Neural Implicit Fields

Code for the final 6.850 project.

The structure is as following:
  * **lsh.py** contains my implementation of the projection-based LSH -- class \textit{L1\_LSH} and the meta class \textit{LSH} that performs parameter search for a given dataset and LSH structure.
  * **LSH\_analysis.ipynb** contains all the code for running analysis of the LSH implementation from the first part.
  * **3D\_shape\_NN.ipynb** contains all the code to query and compare different methods of the mesh representations from the part 2.
  * **WE\_processing.ipynb** contains the code for running and extracting features for the Weight Encoded method. It had to be separated as it requires a separate python environment with tensorflow.
