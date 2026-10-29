import math
from collections import Counter
import numpy as np

def calculate_entropy(labels: list) -> float:
    """Calculate the entropy of a list of labels."""
    if len(labels) == 0:
        return 0.0
        
    labels = np.asarray(labels)
    
    encoded = Counter(labels)

    entropy = 0.0

    for k, v in encoded.items():
        p = v / len(labels)

        entropy += p * np.log2(p)
    
    return -entropy

def calculate_information_gain(examples: list[dict], attr: str, target_attr: str) -> float:
    """Calculate the information gain of splitting on attr."""
    unq_values = set()
    for ex in examples:
        unq_values.add(ex[attr])
    unq_values = list(unq_values)

    labels = []
    for ex in examples:
        labels.append(ex[target_attr])
    result = calculate_entropy(labels)

    for unq_value in unq_values:
        subset_vals = []
        for ex in examples:
            if ex[attr] == unq_value:
                subset_vals.append(ex[target_attr])
        H_v = calculate_entropy(subset_vals)

        result -= len(subset_vals) / len(labels) * H_v
    
    return result

def all_same_class(examples, target_attr):
    cand = examples[0][target_attr]
    for ex in examples:
        if cand != ex[target_attr]:
            return False

    return True

def unique_values(examples, attr):
    result = set()
    for ex in examples:
        result.add(ex[attr])

    return sorted(list(result))


def majority_class(examples: list[dict], target_attr: str) -> str:
    """Return the majority class. Break ties alphabetically."""
    classes = []
    for ex in examples:
        classes.append(ex[target_attr])
    
    pairs = sorted(Counter(classes).items(), key=lambda x: (-x[1], x[0]))

    return pairs[0][0]

def learn_decision_tree(examples: list[dict], attributes: list[str], target_attr: str) -> dict:
    """Build a decision tree using the ID3 algorithm."""
    if all_same_class(examples, target_attr):
        return examples[0][target_attr]
    if not attributes:
        return majority_class(examples, target_attr)

    best_attr = max(attributes, key=lambda x: calculate_information_gain(examples, x, target_attr))

    best = {best_attr : {}}
    for val in unique_values(examples, best_attr):
        subset = [ex for ex in examples if ex[best_attr] == val]
        remaining = [a for a in attributes if a != best_attr]
        best[best_attr][val] = learn_decision_tree(subset, remaining, target_attr)

    return best

    


















    