# learn AIDM
Python AIDM_learner.py 1 state_transitions   > AIDM_logs.txt

# estimate missing action labels
Python ActionEstimator.py Flocking  state_transitions
Python ActionEstimator.py 4G_light  state_transitions

# learn and evaluate the performance of behaviour imitation
Python Imitation_learner.py Flocking  state_transitions LfO  1 > LfO_evaluation_results_Flocking.txt
Python Imitation_learner.py 4G_light  state_transitions LfO  1 > LfO_evaluation_results_4G.txt





