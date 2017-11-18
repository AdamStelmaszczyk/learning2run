from baselines.pposgd.pposgd_simple import OBSERVATION_DIM, set_obstacles

# obstacle1.x = 1
# obstacle2.x = 2.5
# obstacle3.x = 2.5 (hidden behind obstacle2)

result = [0] * 42
dx = 0
step = 1
set_obstacles(result, dx, step)
assert set_obstacles.prev == [-2.0, 0.0, 0.1]
assert set_obstacles.next == [2.0, 0.0, 0.1]
assert result[2:36] == [0] * 34
assert result[36:42] == [-0.4, 0.0, 0.0, 0.4, 0.0, 0.0]

step = 2
result = [0] * 42
result[36] = 1
result[37] = 0.1
result[38] = 0.2
set_obstacles(result, dx, step)
assert set_obstacles.prev == [-2.0, 0.0, 0.1]
assert set_obstacles.next == [1.0, 0.1, 0.2]
assert result[2:36] == [0] * 34
assert result[36:42] == [-0.4, 0.0, 0.0, 0.2, 0.4, 0.1]

step = 3
dx = 0.1
result = [0] * 42
result[1] = 0.1
result[36] = 0.9
result[37] = 0.1
result[38] = 0.2
set_obstacles(result, dx, step)
assert set_obstacles.prev == [-2.1, 0.0, 0.1]
assert set_obstacles.next == [0.9, 0.1, 0.2]
assert result[2:36] == [0] * 34
assert result[36:42] == [-0.4, 0.0, 0.0, 0.18, 0.4, 0.1]

step = 4
dx = 1.1
result = [0] * 42
result[1] = 1.2
result[36] = -0.2
result[37] = 0.1
result[38] = 0.2
set_obstacles(result, dx, step)
assert set_obstacles.prev == [-3.2, 0.0, 0.1]
assert set_obstacles.next == [-0.20000000000000007, 0.1, 0.2]
assert result[2:36] == [0] * 34
assert result[36:42] == [-0.4, 0.0, 0.0, -0.040000000000000015, 0.4, 0.1]

step = 5
dx = 0.1
result = [0] * 42
result[1] = 1.3
result[36] = 1.2
result[37] = 0.3
result[38] = 0.4
set_obstacles(result, dx, step)
assert set_obstacles.prev == [-0.30000000000000004, 0.1, 0.2]
assert set_obstacles.next == [1.2, 0.3, 0.4]
assert result[2:36] == [0] * 34
assert result[36:42] == [-0.06000000000000001, 0.4, 0.1, 0.24, 1.2, 0.30000000000000004]

step = 6
dx = 1.6
result = [0] * 42
result[1] = 2.9
result[36] = 100
result[37] = 0
result[38] = 0
set_obstacles(result, dx, step)
assert set_obstacles.prev == [-0.40000000000000013, 0.3, 0.4]
assert set_obstacles.next == [-2.0, 0.0, 0.0]
assert result[2:36] == [0] * 34
assert result[36:42] == [-0.08000000000000003, 1.2, 0.30000000000000004, -0.4, 0.0, -0.1]
