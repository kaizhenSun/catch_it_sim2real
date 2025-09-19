## This repository is used for deploying the "Catch It" task.

### 重置仿真环境
```shell
rosservice call /reset_sim "reset_request: True"
```
### 重置真实环境
```shell
rosservice call /reset_real "reset_request: True"
```