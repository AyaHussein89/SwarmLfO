# SwarmLfO
Code for swarm learning from observations, as appears in Hussein et al. (2023) "Imitation Learning from Observations in Swarm Systems".

Dependencies:
- pandas
- torch
- sklearn
- pickle
- cv2

Example usage:
1. Bash explore.sh: run the exploration phase 
2. Bash generateDemos.sh:  generate the expert's observations/demonstrations for both the flocking and sheltering ("4G_light") tasks
3. Bash evaluateExpert.sh: get evaluation results for expert performance for the flocking and sheltering ("4G_light") tasks
4. LfORuns.sh: must be used after the exploration phase is complete (1.) and the expert observations are provided (2.). It learns the AIDM, estimate missing actions, and then train and evaluate the behaviour imitation performance for both tasks
5. DecLfORuns.sh: must be used after the exploration phase is complete (1.) and the expert observations are provided (2.). Used to generate the results for Dec-Exp-LfO  in the paper
6. LfDRuns.sh:   must be used after the expert demonstrations are provided (2.) Used to generate the results for LfD model in the paper. 




