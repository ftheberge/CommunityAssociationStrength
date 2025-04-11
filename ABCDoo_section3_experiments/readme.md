Each notebook generates figures for a single experiment.
There are some additional scripts, file, and notebooks for auxilliary functions.
Follow the instructions below to reproduce figures.
*Make sure each of the scripts is run from inside the ```ABCDoo_section3_experiments``` folder so local file paths function properly*

## Section 3
To get started, ensure there is a folder titled ```data```.
In this folder put the DBLP, AMAZON and YOUTUBE graph and community files from the [SNAP](https://snap.stanford.edu/data/#communities) website.
The folder should look like:

data/
- com-amazon.all.dedup.cmty.txt
- com-amazon.ungraph.txt
- com-dblp.all.cmty.txt
- com-dblp.ungraph.txt
- com-youtube.all.cmty.txt
- com-youtube.ungraph.txt

Next, run the julia file ```abcdoo_snap_graph_sampler.jl```.
This will generate ABCD+O^2 graphs using parameters measured from the real graphs and save to the data folder.
The parameters are hard-coded into the julia file.
We measured the parameters using ```measure_params.ipynb```.

At this point the majority of the experiments can be run.
Below are the notebooks listed in the order that the figures they produce appear.

Experiments in Section 3.1
- ```community_size.ipynb```
- ```communities_per_node.ipynb```
- ```community_overlap.ipynb```

Experiments in Section 3.2
- ```correlation.ipynb```
- ```degree.ipynb```
- ```overlap_density.ipynb```
- ```CAS_strength.ipynb```

The experiment compaing rho to overlap density requires new data.
Run ```abcdoo_rho_sampler.jl``` (again the parameters have been hard-coded).
Then use the notebook ```rho_vs_overlap_density.ipynb```.


## Section 4