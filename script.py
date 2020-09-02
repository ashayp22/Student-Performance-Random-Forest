import math
from collections import *
from functools import *
import random
import csv


inputs = []

remove = ['nationality', 'placeofbirth', 'sectionid', 'topic', 'semester', 'relation']
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

print(inputs)

#helper methods

def entropy(class_probabilities):
    return sum(-p * math.log(p, 2)
               for p in class_probabilities
               if p)

def class_probabilities(labels):
    total_count = len(labels)
    return [count / total_count
            for count in Counter(labels).values()]

def data_entropy(labeled_data):
    labels = [label for _, label in labeled_data]
    probabilities = class_probabilities(labels)
    return entropy(probabilities)

def partition_entropy(subsets):

    total_count = sum(len(subset) for subset in subsets)

    return sum( data_entropy(subset) * len(subset) / total_count
                for subset in subsets)

def partition_by(inputs, attribute):
    groups = defaultdict(list)
    for input in inputs:
        key = input[0][attribute]
        groups[key].append(input)
    return groups

def partition_entropy_by(inputs, attribute):
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

def build_tree_id3(inputs, num_split_candidates=2, split_candidates=None):

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

    subtrees = { attribute_value : build_tree_id3(subset, split_candidates=new_candidates)
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
    return vote_counts.most_common(1)[0][0]


data_1 = {'gender': 'M', 'stageid': 'MiddleSchool', 'raisedhands': 'L', 'visitedresources': 'L', 'announcementsview': 'L', 'discussion': 'L', 'parentansweringsurvey': 'Yes', 'parentschoolsatisfaction': 'Good', 'studentabsencedays': 'Under-7'}
data_2 = {'visitedresources': 'H', 'announcementsview': 'H', 'discussion': 'H', 'parentansweringsurvey': 'Yes', 'parentschoolsatisfaction': 'Good', 'studentabsencedays': 'Under-7'}


all_trees = [build_tree_id3(random.sample(inputs, 100)) for _ in range(50)]

print(forest_classify(all_trees, data_1))
print(forest_classify(all_trees, data_2))

good = 0
total = 0

for i in inputs:
    c = forest_classify(all_trees, i[0])
    if c == i[1]:
        good += 1
    total += 1

print(good / total)
