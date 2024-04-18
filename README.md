# Packages

You can use the `environment.yml` file, BUT  it is only for CPU.

openmm, dmff, mdtraj

openmm needs to be installed by conda following the instruction

mdtraj is easy to install 

dmff needs to be installed from source code https://github.com/deepmodeling/DMFF/blob/master/docs/user_guide/2.installation.md
you can install it with 

```
pip install dmff @ git+https://github.com/deepmodeling/DMFF@v1.0.0
```

# Run the code 
```
python main_md.py
```
