import torch
import pathlib
import avg_index

# You can run these test with `pytest` on the command line.


def test_average_equals_single_input(tmpdir: pathlib.Path):
    xx = [torch.tensor(x) for x in [1.0, 2.0, 3.0]]

    index = avg_index.UniformAvgIndex(tmpdir)
    index.add(xx)

    for avg, x in zip(index.current_avg(), xx):
        assert torch.allclose(avg, x)


def test_average_is_correct_at_t_2(tmpdir: pathlib.Path):
    xx = [torch.tensor(x) for x in [1.0, 2.0, 3.0]]
    yy = [torch.tensor(x) for x in [2.0, 3.0, 4.0]]
    expected_avg = [torch.tensor(x) for x in [1.5, 2.5, 3.5]]

    index = avg_index.UniformAvgIndex(tmpdir)
    index.add(xx)
    index.add(yy)

    for avg, z in zip(index.current_avg(), expected_avg):
        assert torch.allclose(avg, z)


def test_automatic_checkpoints_work(tmpdir: pathlib.Path):
    index = avg_index.UniformAvgIndex(tmpdir, checkpoint_period=2)
    for _ in range(6):
        index.add([torch.ones(2)])

    assert index.available_checkpoints == {2, 4, 6}


def test_manual_checkpoints_work(tmpdir: pathlib.Path):
    index = avg_index.UniformAvgIndex(tmpdir)
    for _ in range(6):
        index.add([torch.ones(2)])
    index.store_checkpoint()
    index.add([torch.ones(2)])
    index.store_checkpoint()

    assert index.available_checkpoints == {6, 7}


def test_window_averages_are_correct(tmpdir: pathlib.Path):
    index = avg_index.UniformAvgIndex(tmpdir, checkpoint_period=1)
    for i in range(10):
        index.add([torch.tensor(float(1 + i))])

    assert torch.allclose(index.avg_from(9, until=10)[0], torch.tensor(10.0))
    assert torch.allclose(index.avg_from(4, until=6)[0], torch.tensor(5.5))
    assert torch.allclose(index.avg_from(8)[0], torch.tensor(9.5))
    assert torch.allclose(index.avg_from(0, until=10)[0], torch.tensor(5.5))


def test_current_avg_is_a_copy(tmpdir: pathlib.Path):
    index = avg_index.UniformAvgIndex(tmpdir, checkpoint_period=1)

    index.add([torch.ones(1)])
    avg = index.current_avg()
    index.add([torch.zeros(1)])

    assert torch.allclose(avg[0], torch.tensor(1.0))
