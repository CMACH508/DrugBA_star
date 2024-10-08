# $DrugBA^{\*}$
This is the open-source codebase the paper: __De Novo Drug Design by Multi-Objective Path Consistency Learning with Beam $A^{\*}$ Search__. Please do not hesitate to contact us if you have any problems.

## Description
Generating high-quality and drug-like molecules from scratch within the expansive chemical space presents a significant challenge in the field of drug discovery. In prior research, value-based reinforcement learning algorithms have been employed to generate molecules with multiple desired properties iteratively. The immediate reward was defined as the evaluation of intermediate-state molecules at each step and the learning objective would be maximizing the expected cumulative evaluation scores for all molecules along the generative path. However, this definition of the reward was misleading, as in reality, the optimization target should be the evaluation score of only the final generated molecule. Furthermore, in previous works, randomness was introduced into the decision-making process, enabling the generation of diverse molecules but no longer pursuing the maximum future rewards. In this paper, immediate reward is defined as the improvement achieved through the modification of the molecule to maximizing the evaluation score of the final generated molecule exclusively. Our contribution can be summarized as follows:

- We propose a novel value-based reinforcement learning algorithm for *de novo* drug design to optimize the final generated molecule exclusively and consider multiple desired properties concurrently. Originating from the $A^{\*}$ search, the path consistency condition, i.e., $f$ values on one optimal path should be identical, is utilized as the learning target to update the state evaluation function $f(s;\theta)$.

-  $DrugBA^{\*}$ search, which is built upon the beam search algorithm, is proposed to generate a batch of molecules efficiently. This approach allows for the simultaneous production of a large number of molecules, facilitating the exploration of chemical space while maintaining the pursuit of optimality. At each step, the best partial solutions are determined based on the $f$ value, inspired by the $A^{\*}$ search algorithm.

- The experimental results demonstrate the effectiveness of $DrugBA^{\*}$ in both single property optimization and multi-objective optimization tasks. $DrugBA^{\*}$ has exhibited significant improvements across all evaluation metrics compared to the state-of-the-art algorithm QADD. 


## Dependency
- python==3.7.15
- torch==1.9.5
- numpy==1.21.5
- pickle==0.0.12
- rdkit==2022.9.3

## Pipeline
Training the $f$ value estimation function:

```
python model.py
```

After obtaining the heuristic $f$ function, molecules are generated by the beamsearch algorithm:

```
python search.py
```

To evaluate the generated molecules, run:

```
python eval.py
```
The calculation of the evaluation scores are the same with QADD. ChEMBL database is required to calculate the novelty score (<https://chembl.gitbook.io/chembl-interface-documentation/downloads>).
## Lisence
- All materials are made available under the terms of the Creative Commons Attribution-NonCommercial 4.0 International Public License (CC BY-NC 4.0) license. You can find details at: <https://creativecommons.org/licenses/by-nc/4.0/legalcode>

- The license gives permission for academic use only.
