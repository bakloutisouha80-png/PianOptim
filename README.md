# PianOptim

An optimal control project to simulate piano

## Getting ready

### bioptim

Either the `bioptim` submodule is expected to appear in `dependencies/bioptim`. If the folder is empty, you should initialize it using git. 

Otherwise, `bioptim` can be installed using `conda` from the `conda-forge` channel. 

### Prepare for vscode

In the `.vscode` folder, copy-paste the `.env.default` file to `.env`. Adjust the separator if neede (":" for UNIX and ";" for Windows). 


## Things to discuss

### What "not using the trunk" actually means?

There are two ways to model this:
1. We can remove the degrees-of-freedom for the trunk. This means it won't be able to move, but the dynamic of the arms is concurently messed up. The reason is any movement of the arm, whatever the internal forces it creates, can be balanced out by this infinitely strong trunk. So it may try to transfer as much generalized forces as possible to the trunk
2. The second method is to constraint the trunk to have a velocity equals to zero at each nodes, effectively nullifying the movement of the trunk. This has the advantage of keeping the dynamic intact, but may be much harder to optimize

- Whatever what is decide, do we prescribe the pose actually held?

### Impact of the finger on the key
- Usually we should call the "IMPACT" phase transition when an impact occurs. Howerver, here, the impact is "smooth". Does that mean no impact phase transition is needed?

  - Solution is to use damping 

### Cost function
- Power seems a good idea, but it is not integrated, meaning a huge torque is not balanced out by a huge penalty for one node if velocity is small, for the hole node. Meaning it can oscillate. Should we use LINEAR_CONTINUOUS? Should we add Power as an algebraic state?

### What is the best way to model the press phase?

- Should we track a speed profile of the key from actual data?
- Should we track a force profile at the finger from actual data?
- Should we model the force from key? Using an exponential to simulate the bed of the key? Free time?
- Should we have a dynamic model of the sound? Artificial intelligence? Free time?

### Fingers are way too light

- This may cause (and probably is causing) problems when inverting the matrix when computing the dynamics (reason why forced to use COLLOCATION?)
- Is this even relevent to keep fingers for the question we are trying to answer?

# Discussion
Comparer simulation tronc VS simulation tronc fixe
Quel est le mécanisme utilisé quand on a beaucoup de DoF
  - Plutôt postural
  - Dépendance de la fonction objectif
  - Quelles sont les coordinations qui permettent une utilisation efficace du tronc

### TODO
11/04/2025

**follow through**
- Avec la main plutot que le bout du doigts
- Poulen de doigts plutot part en avant (s'éloigne du joueur) a la fin de la pression sur la touche plutot qu'en arriere
  - mettre une condition que le centre de masse de la main avance pendant la descente, puis main + bras, puis main + bras + thorax
  - Faire l'analyse des resultats la dessus

**link stomach and pelvis** 
Motion to be in the same direction. q sign should always be the same

**Fonction de cout** 
Taudot pas minimser pour le coude poignet et epaule (a faire pour epaule) !

**Pelvis torques**
- Opposite torque from leg weight.
    - 0.8 * 0.15 * 70 * 9.81 = 82.404 Nm estimation couple max du pelvis (rotation arriere), provient de l'appui des pieds
    - 0.8 BW force from sit to stand motion https://doi.org/10.1016/j.archger.2007.10.006
- Forward torque from leg weight and inertia

**Contact**
- spring-damper for the key, integrer les resultats identifier sur le ressort (regression polynomial et exponentielle)
- Friction cone for the finger to prevent sliding (as in somersault simulations)
