# This class is responsible of making a real test scenario of amazon reviews
# The training is completely balanced from the training.csv file
# The test set will be taken from the test_unbalanced.csv file and redefined like a real scenario

# The real test dataset will be:
# 80 samples of 1 star reviews
# 40 samples of 2 star reviews
# 40 samples of 3 star reviews
# 100 samples of 4 star reviews
# 300 samples of 5 star reviews

# the new test set will be saved in the dataset folder as test_real_scenario.csv

import pandas as pd

def real_test_creation(data):
    # load test_unbalanced.csv
    test_unbalanced = pd.read_csv(data)
    # create the real test dataset
    test_real_scenario = pd.DataFrame(columns=['reviewText', 'overall'])
    # get the reviews
    reviews = test_unbalanced['reviewText']
    # get the overall
    overall = test_unbalanced['overall']
    
    # insert 1 star reviews
    count = 0
    for i in range(len(reviews)):
        if overall[i] == 1:
            test_real_scenario = test_real_scenario._append({'reviewText': reviews[i], 'overall': 1}, ignore_index=True)
            count += 1
            if count == 30:
                break

    # insert 2 star reviews
    count = 0
    for i in range(len(reviews)):
        if overall[i] == 2:
            test_real_scenario = test_real_scenario._append({'reviewText': reviews[i], 'overall': 2}, ignore_index=True)
            count += 1
            if count == 15:
                break

    # insert 3 star reviews
    count = 0
    for i in range(len(reviews)):
        if overall[i] == 3:
            test_real_scenario = test_real_scenario._append({'reviewText': reviews[i], 'overall': 3}, ignore_index=True)
            count += 1
            if count == 20:
                break

    # insert 4 star reviews
    count = 0
    for i in range(len(reviews)):
        if overall[i] == 4:
            test_real_scenario = test_real_scenario._append({'reviewText': reviews[i], 'overall': 4}, ignore_index=True)
            count += 1
            if count == 50:
                break
    # insert 5 star reviews
    count = 0
    for i in range(len(reviews)):
        if overall[i] == 5:
            test_real_scenario = test_real_scenario._append({'reviewText': reviews[i], 'overall': 5}, ignore_index=True)
            count += 1
            if count == 500:
                break
    # save the new test set
    test_real_scenario.to_csv('dataset/test_real_scenario.csv', index=False)

if __name__ == '__main__':
    real_test_creation('dataset/test_unbalanced.csv')
