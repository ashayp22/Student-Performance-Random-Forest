import math
from collections import *
from functools import *
import random
import csv

#source: "Data Science from Scratch" by Joel Grus

inputs = []

remove = ['gender', 'nationality', 'placeofbirth', 'sectionid', 'topic', 'semester', 'relation', 'parentschoolsatisfaction']
convert = ['raisedhands', 'visitedresources', 'announcementsview', 'discussion']

with open('api-edu-data.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    keys = []
    for row in csv_reader:
        if line_count == 0:
            keys = [x.lower() for x in row]
            print(keys)
        else:
            new_dict = {}
            c = 0
            for key in keys:
                if key not in remove: #should be added to the dict
                    if key in convert: #needs to be convereted from numerical data to categorical data
                        val = int(row[c])
                        m = ""
                        if val <= 33:
                            m = "L"
                        elif val <= 66:
                            m = "M"
                        else:
                            m = "H"
                        new_dict[key] = m
                    else:
                        new_dict[key] = row[c]
                c += 1

            grade = new_dict['class'] == "H"
            new_dict.pop('class')
            inputs.append((new_dict, grade))
        line_count += 1

# print(inputs)

#helper methods


def entropy(class_probabilities):
    """given the class probabilities as a list, we can compute the entropy of the dataset"""
    return sum(-p * math.log(p, 2)
               for p in class_probabilities
               if p)

def class_probabilities(labels):
    '''this returns our probabilites'''
    total_count = len(labels)
    return [count / total_count
            for count in Counter(labels).values()]

def data_entropy(labeled_data):
    '''labeled_data is inputted as (input, label), but we only care about the input'''
    labels = [label for _, label in labeled_data]
    probabilities = class_probabilities(labels)
    return entropy(probabilities)

def partition_entropy(subsets):
    '''finds the entropy from all of the partitions of data'''
    total_count = sum(len(subset) for subset in subsets)

    return sum( data_entropy(subset) * len(subset) / total_count
                for subset in subsets)

def partition_by(inputs, attribute):
    """each input is a pair (attribute, label)
    this returns a dict : attribute -> inputs"""
    groups = defaultdict(list)
    for input in inputs:
        key = input[0][attribute]
        groups[key].append(input)
    return groups

def partition_entropy_by(inputs, attribute):
    '''computes the entropy corresponding to the given partition'''
    partitions = partition_by(inputs, attribute)
    return partition_entropy(partitions.values())

def classify(tree, input):

    if tree in [True, False]:
        return tree

    attribute, subtree_dict = tree

    subtree_key = input.get(attribute)

    if subtree_key not in subtree_dict:
        subtree_key = None

    subtree = subtree_dict[subtree_key]
    return classify(subtree, input)

def build_tree(inputs, num_split_candidates=2, split_candidates=None):

    if split_candidates is None:
        split_candidates = inputs[0][0].keys()

    #count the Trues and Falses in the inputs
    num_inputs = len(inputs)
    num_trues = len([label for item, label in inputs if label])
    num_falses = num_inputs - num_trues

    if num_trues == 0: return False #no trues, return False
    if num_falses == 0: return True #no false, return True

    if not split_candidates: #if not split candidate left, return the majority leaf
        return num_trues >= num_falses

    # add in the random forest part

    if len(split_candidates) <= num_split_candidates:
        sampled_split_candidates = split_candidates
    else:
        sampled_split_candidates = random.sample(split_candidates, num_split_candidates)

    #otherwise, split on the best candidate
    best_attribute = min(sampled_split_candidates, key=partial(partition_entropy_by, inputs))

    partitions = partition_by(inputs, best_attribute)
    new_candidates = [a for a in split_candidates
                      if a != best_attribute]

    #recursively buld the subtrees

    subtrees = { attribute_value : build_tree(subset, split_candidates=new_candidates)
                 for attribute_value, subset in partitions.items() }

    subtrees[None] = num_trues > num_falses #default case

    return (best_attribute, subtrees)


'''
To prevent overfitting, we can build random forests in which we
build multiple decision trees and let them vote on how to classify
inputs.
'''

#random forest technique 1 - build multiple decision trees and let them vote on how to classify inputs
def forest_classify(trees, input):
    votes = [classify(tree, input) for tree in trees]
    vote_counts = Counter(votes)

    if vote_counts.most_common(1)[0][0]:
        return "High Performance"

    return "Low Performance"

testing = []

for _ in range(15):
    index = random.randint(0, len(inputs)-1)
    testing.append(inputs.pop(index))

all_trees = [build_tree(random.sample(inputs, 200)) for _ in range(50)]

student_1 = {'stageid': 'MiddleSchool', 'raisedhands': 'L', 'visitedresources': 'M', 'announcementsview': 'L', 'discussion': 'L', 'parentansweringsurvey': 'Yes', 'studentabsencedays': 'Over-7'}
student_2 = {'visitedresources': 'H', 'announcementsview': 'H', 'discussion': 'H', 'parentansweringsurvey': 'Yes', 'studentabsencedays': 'Under-7'}


print("Student 1 has " + forest_classify(all_trees, student_1))
print("Student 2 has " + forest_classify(all_trees, student_2))

good = 0
total = 0

for i in testing:
    c = forest_classify(all_trees, i[0])
    i_class = "Low Performance"
    if i[1]:
        i_class = "High Performance"
    if c == i_class:
        good += 1
    total += 1

print("The random forest's accuracy is " + str(good / total * 100) + "%")
