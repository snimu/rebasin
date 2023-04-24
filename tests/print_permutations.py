import torch

from rebasin import PermutationCoordinateDescent
from tests.fixtures.models import MLP

if __name__ == '__main__':
    model_a = MLP(25, num_layers=10)
    model_b = MLP(25, num_layers=10)
    pcd = PermutationCoordinateDescent(model_a, model_b)
    pcd.calculate_permutations()

    print(len(pcd.permutations))
    for permutation in pcd.permutations:
        print(permutation)
