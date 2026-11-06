# 复现清单

- [x] 固定来源六D501完整checkpoint，source_state_audit含SHA/Adam10020/实际LR/RNG。
- [x] train方法/seed/map/order与eval矩阵结果前定义；train/eval/R maps隔离，保留完整hfield数组。
- [x] Python/NumPy/CPU-CUDA RNG在全部构造后从D恢复；同phase跨域精确配对。
- [x] 只使用冻结SourceAnchoredPPO beta.1/原teacher/98actor/原critic/reward；完整warm start，新仿真非精确续接。
- [x] 保存native20旧KL+20sourceKL、Adam累计步、固定checkpoint和ONNX SHA，逐step有限/容量，边界真实hfield接触。
- [x] raw与物理/初态/command/首次native终止/完整20sMAE/8family/dualclock和flat保护独立CPU复算脚本结果前封存。
- [ ] 最终代码CPU preflight、fresh实现review与协议封存（运行后填工程进度，不回写本冻结清单）。
- [ ] 真实smoke通过后才唯一正式queue；完成后480raw独立审计。

Environment: 现有Windows Python3.13/CUDA/torch/mjlab/MuJoCo-Warp/rsl_rl；版本由protocol锁定，不修改site-packages。Paths和完整依赖SHA由S协议source_snapshot保存。0采样工程检查不能写成训练成功；同操作者独立计算不是第二reviewer。

Determinism: fresh child启动PYTHONHASHSEED=20264200，环境构造按来源seed，首次采样前恢复源Python/NumPy/torch/CUDA RNG。cuDNN deterministic=True、benchmark=False、CUDA matmul TF32=False（a.setup已设置）；逐源checkpoint0/RNG精确核验限制是本机同软件栈，不承诺跨硬件轨迹相同。训练和在线推断无DataLoader，不存在worker随机流；若未来加入DataLoader必须前锁worker_init_fn并重新审核，当前不擅自添加。
