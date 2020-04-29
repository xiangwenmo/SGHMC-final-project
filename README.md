# SGHMC-final-project
Implement Stochastic Gradient Hamilton Monte Carlo  
Final project for STA-663, by Xiangwen Mo(xiangwen.mo@duke.edu), Bingruo Wu(bingruo.wu@duke.edu)

Details about SGHMC are in [paper by Tianqi Chen Emily B. Fox Carlos Guestrin](https://arxiv.org/pdf/1402.4102.pdf)

### Install and import package

run

```
!pip install --index-url https://test.pypi.org/simple/ xmbwsghmc==0.0.2
from sghmc import sghmc_algo
```

### Contents

- `sghmc/` : source codes for the package  

- `examples/`: two simulation examples and a attempt to implement on real datasets  

- `opt:comparison/`: optimization and comparing algorithm with `pystan` and `pyhmc`  

- `report/`: project report
