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


def test_module_does_not_crash(tmpdir: pathlib.Path):
    model = torch.nn.Sequential(
        torch.nn.Linear(1, 1),
        torch.nn.BatchNorm1d(1),
    )
    index = avg_index.ModelAvgIndex(
        model,
        avg_index.UniformAvgIndex(tmpdir, checkpoint_period=1),
        include_buffers=True,
    )

    index.record_step()
    index.avg_from(0)


def test_averaging_changes_the_module(tmpdir: pathlib.Path):
    model = torch.nn.Sequential(
        torch.nn.Linear(3, 3),
        torch.nn.BatchNorm1d(3),
    )
    index = avg_index.ModelAvgIndex(
        model,
        avg_index.UniformAvgIndex(tmpdir, checkpoint_period=1),
        include_buffers=True,
    )

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    for _ in range(2):
        output = model(torch.randn([4, 3]))
        output.mean().backward()
        optimizer.step()
        index.record_step()

    input = torch.randn([4, 3])
    model.training = False
    avg_module = index.avg_from(0)
    out_vanilla = model(input)
    out_avg = avg_module(input)

    assert not torch.allclose(out_vanilla, out_avg)


def test_t_avg(tmpdir: pathlib.Path):

    index1 = avg_index.TriangleAvgIndex(tmpdir, checkpoint_period=1)
    for i in range(10):
        index1.add([torch.tensor(i + 1).float()])

    index2 = avg_index.TriangleAvgIndex(tmpdir, checkpoint_period=1)
    for i in range(5, 10):
        index2.add([torch.tensor(i + 1).float()])

    assert torch.allclose(index2.avg_from(0)[0], torch.tensor(8.6666667))
    assert torch.allclose(index1.avg_from(0, until=5)[0], torch.tensor(3.6666667))
    assert torch.allclose(index1.avg_from(5, until=10)[0], torch.tensor(8.6666667))

def test_state_dict(tmpdir: pathlib.Path):
    model = torch.nn.Sequential(
        torch.nn.Linear(3, 3),
        torch.nn.BatchNorm1d(3),
    )
    u_index = avg_index.UniformAvgIndex(tmpdir, checkpoint_period=1)
    u_index._available_checkpoints = set([0, 1, 2])
    u_index._counter = 4
    u_index._current_avg = list(model.parameters())

    sd = u_index.state_dict()
    assert sd['available_checkpoints'] == u_index._available_checkpoints
    assert sd['checkpoint_dir'] == u_index._checkpoint_dir
    assert sd['checkpoint_period'] == u_index._checkpoint_period
    assert sd['counter'] == u_index._counter
    assert sd['current_avg'] == u_index._current_avg
    assert sd['uuid'] == u_index._uuid

    new_u_index = avg_index.UniformAvgIndex(tmpdir, checkpoint_period=10)
    new_u_index.load_state_dict(sd)
    assert sd['available_checkpoints'] == new_u_index._available_checkpoints
    assert sd['checkpoint_dir'] == new_u_index._checkpoint_dir
    assert sd['checkpoint_period'] == new_u_index._checkpoint_period
    assert sd['counter'] == new_u_index._counter
    assert sd['current_avg'] == new_u_index._current_avg
    assert sd['uuid'] == new_u_index._uuid
