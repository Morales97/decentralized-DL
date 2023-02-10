import torch

import running_mean_and_var


def test_mean_is_correct():
    mv = running_mean_and_var.RunningMeanAndVar()
    numbers = torch.arange(10).float()

    for number in numbers:
        mv.add(number)

    gt_var, gt_mean = torch.var_mean(numbers, unbiased=False)
    gt_var_unbiased = torch.var(numbers, unbiased=True)
    assert mv.mean == gt_mean
    assert mv.sample_variance() == gt_var
    assert mv.sample_variance(unbiased=True) == gt_var_unbiased


def test_empirical_mean_variance_is_correct():
    mv = running_mean_and_var.RunningMeanAndVar()
    numbers = torch.arange(10).float()

    for number in numbers:
        mv.add(number)

    assert mv.sample_variance(unbiased=True) / 10 == mv.variance_of_mean()


def test_add_many_works():
    numbers = torch.arange(4).float()

    mv1 = running_mean_and_var.RunningMeanAndVar()
    for number in numbers:
        mv1.add(number)

    mv2 = running_mean_and_var.RunningMeanAndVar()
    mv2.add_many(numbers)

    assert mv1.mean == mv2.mean
    assert mv1.sample_variance() == mv2.sample_variance()
