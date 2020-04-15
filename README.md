This code is implemented according to paper "Deep Bayesian Optimization on Attributed graphs",
https://www.aaai.org/ojs/index.php/AAAI/article/view/3938/3816. Deep Graph Bayesian Optimization (DGBO) method can deal with 
attributed graphs. It prevents the cubical complexity of the GPs by adopting a deep graph neural 
network to surrogate black-box functions, and can scale linearly with the number of observations.
Applications include molecular discovery and urban road network design.

If you want to run this code, you should ensure that the following packages have been installed 
successfully:

    tensorFlow
    spicy
    pickle
    numpy
    emcee
    networkx

Once you install all dependency packages, you can run this code with the default setting as:

$$ python DGBO.py --run=True

, or you can see the help message by running as:

$$ python DGBO.py -h

Note: If you try the zinc dataset, you should run “genConvMolFeatures.py” in “rdkit_preprocessing/” 
to convert SMILES strings to attributed graphs including xxx-attr.pkl, xxx-graph.pkl, and xxx-label.pkl.
