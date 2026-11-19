# 网页关节监控

网页及Python后端均已收进本仓库，无需另一个unitree_mujoco源码仓库。
监控只订阅DDS，不发布LowCmd、不释放运动服务，也不提供运动控制按钮。
它显示 29 个关节的角度、速度、反馈力矩、目标、误差和 IMU，以及最近 20 秒的实时曲线。
后端同时保存每次运行的采样，网页可按时间回看和下载。
实际目标与空跑目标分开处理；空跑时显示`OFF · 空跑`。

## 运行依赖

C++控制程序仍不依赖Python。仅网页需要Linux上的Python 3.10或更新版本、
CPU架构匹配的CycloneDDS C运行库、Python依赖，以及官方`unitree_sdk2py`包。
本次机器人验证环境为Ubuntu 22.04 / aarch64 / Python 3.10 / CycloneDDS Python 0.10.2。
SDK及CycloneDDS运行库和控制程序使用同一套，避免混用ROS环境中的另一套DDS库。

如果Python已经能导入`unitree_sdk2py`和`cyclonedds`，可直接运行。
否则，在仓库根目录安装到项目内；将下面的SDK源码路径替换为本机官方Python SDK路径：

```bash
CYCLONEDDS_HOME=/usr/local python3 -m pip install --target .python-deps -r apps/monitor/requirements.txt
python3 -m pip install --no-deps --target .python-deps /path/to/unitree_sdk2_python
```

官方Python SDK来源：https://github.com/unitreerobotics/unitree_sdk2_python 。
`--no-deps`避免安装本监控不使用的摄像头依赖，所需DDS和NumPy已在上一步安装。
`.python-deps/`是本机依赖目录，不提交Git。若使用虚拟环境，可通过`PYTHON_BIN`指定解释器。

## 从源码目录启动

```bash
cd /home/unitree/g1_deploy
bash apps/monitor/monitor.sh enP8p1s0 192.168.123.164 8765
```

在同一网络的电脑浏览器访问`http://192.168.123.164:8765/`。
第一个参数是实际DDS网卡，第二个是HTTP监听地址，第三个是端口。
不指定地址时仅监听 `127.0.0.1`。真机通信域为 0，网页实时视图最高 50 Hz。
按Ctrl+C停止网页，不影响其他控制程序。不设置开机自启。

也可直接启动Python后端：

```bash
python3 apps/monitor/joint_monitor.py --domain 0 --interface enP8p1s0 \
    --host 192.168.123.164 --port 8765 --rate 50 --record-rate 250
```

直接运行Python时，需要已安装依赖或自行将`.python-deps`加入`PYTHONPATH`。

## 运行记录

先启动监控后端，再启动控制程序。收到本项目控制器的 Enable 状态后自动开档，并写入此前最多 2 秒的缓存。实机接管后按空格或 Select 退出时，记录保存到停止输出为止。控制状态断流 3 秒后自动结束。快速重启时使用 DDS 发布者标识分段。缺少该标识的兼容环境中，短时间内连续的纯空跑可能合并。内嵌的 MuJoCo 网页按接管到停止分段。

网页顶部选择一次运行后，可拖动时间轴、勾选任意关节，并在 1、5、20 秒的窗口内查看曲线。“下载 29 关节 CSV”导出该次运行的全部已写入样本，每行对应一个采样时刻的一个关节。暂停显示、刷新或关闭浏览器期间，后端继续记录。

- 实机后端默认记录上限为 250 Hz，直接启动 Python 时可用 `--record-rate` 调到最高 500 Hz。实际采样率取决于 DDS 收包和系统调度。内嵌 MuJoCo 网页最高 50 Hz。
- 每帧保存单调递增的运行时间、本机 Unix 时间、关节角度和速度、估计反馈力矩 `tau_est`、目标与增益，以及最新可用的 `LowCmd` 字段和接收年龄。CSV 中的 `tau` 是反馈值，`tau_cmd` 是指令字段，使用驱动侧 PD 时该字段只代表前馈力矩。DDS 状态与命令按监控端接收时间配对，并保存可用的 DDS 源时间戳和指令发布者标识。过期命令和无效测量留空。
- `model_test` 默认 `drive_pd`，实机对照优先看 CSV 的 `tau` 和 `kp_cmd/kd_cmd`，`tau_cmd` 为零前馈。使用 `--actuation host_torque` 时，`tau_cmd` 记录机载程序下发的限幅力矩。MuJoCo 的 `tau` 是模拟施加的力矩，运行名称包含所选执行模式。
- 数据库默认位于 `$XDG_STATE_HOME/g1_deploy/joint_runs.sqlite3`，未设置时使用 `~/.local/state/g1_deploy/joint_runs.sqlite3`。可用环境变量 `G1_JOINT_HISTORY` 或 Python 参数 `--history-db` 自定义路径。监控重启后仍可查看原有记录，磁盘占用随保存的运行增加。

## 安装目录

`cmake --install build`会同时安装`apps/monitor/`和`monitor`启动脚本：

```bash
cd install
./monitor enP8p1s0 192.168.123.164 8765
# 等效入口：./g1 monitor enP8p1s0 192.168.123.164 8765
```

Python依赖不打进CMake安装包，需由目标机器环境提供，或安装到安装目录的`.python-deps/`。
网页静态文件按Python源码位置加载，目录整体移动后仍能使用。

## 接口

- `/`：网页。
- `/api/joints`：当前29关节、IMU、数据年龄、输出和空跑状态。
- `/api/stream`：SSE实时数据流。
- `/api/runs` 返回运行列表和记录状态。
- `/api/runs/<id>/window?end=5&seconds=1&joints=0,3` 按运行相对时间返回最新状态和指定 DDS 关节的曲线。
- `/api/runs/<id>/joints.csv` 下载整次运行的关节样本。

监控的 Enable 表示本控制程序的输出状态。机器人电源和其他服务的状态需从对应接口读取。
