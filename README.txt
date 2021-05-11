Welcome to our CS 182 Computer Vision Project! We have developed a Computer Vision model that is robust against adversarial data perturbations.

In order to run our code, please perform the following steps:

  1. cd data/
  2. ./get_data.sh (If this results in a Permission denied error, please run 'chmod +x get_data.sh' and then run './get_data.sh' again)
  3. cd ..
  4. pip install -r requirements.txt
  5. Run python3 test_submission.py eval.csv
  
Our test_submission.py file is configured to use GPU 0.
  
