<p align="center">
    <img width="100" src="/Assets/rlai.png" />
</p>

<h2 align=center>Off-policy Prediction Learning Algorithms</h2>
<div align="center">
  :steam_locomotive::train::train::train::train::train:
</div>
This repository ... 


<p align="center">
    <img src="/Assets/fourRoomGridWorld.gif" />
</p>

## Table of Contents
- **[Algorithms](#algorithms)**: [Off-policy TD](#td), [GTD](#gtd), [Emphatic TD](#Emphatic_TD)
- **[Environment](#environment)** : [Four Room Grid World](#four_room_grid_world), [Chain](#chain)

## Run
- [Learning.py](#learning.py)
- [Job Buidler](#job_builder)




<a name='algorithms'></a>
## Algorithms
<hr>
<a name='td'></a>

### Off-policy TD

**Paper** [Off-Policy Temporal-Difference Learning with Function Approximation](https://www.cs.mcgill.ca/~dprecup/publications/PSD-01.pdf)<br>
**Author** Doina Precup, Richard S. Sutton, Sanjoy Dasgupta<br>

#### Main update rule:
```python
def learn_wights(s, s_p, r):
        delta = compute_delta(s, s_p, r, gamma)
        w += alpha * delta * z
```
where s and s_p are the current and next states, r is the reward, and gamma is the discount factor parameter
<hr>


<a name='environment'></a>
## Environment
<hr>
<a name="four_room_grid_world"></a>

### Four Room Grid World

<a name="four_room_grid_world"></a>

### Chain

## Run
<hr>
<a name="learning.py"></a>
### Learning.py
```sh
$ learning.py -p1 p1
```

<a name="job_builder"></a>
### Job Builder


