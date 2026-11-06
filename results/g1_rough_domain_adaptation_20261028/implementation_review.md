# 技术修订01 fresh定点复核

Reviewer: /root/rough_domain_implementation_review。未解决Critical/Important/Minor均0；在最新preflight与修订源码SHA绑定重新封存后允许完整固定smoke，不代表GPU通过。

首smoke在完整source恢复后、0控制步因原生35个未命名robot geom被名称检查拒绝。technical_failure_01保存原协议/源码/日志/失败manifest/进程资源和成本；0train/eval/PPO/Adam。technical_amendment_01已预锁不改模型/预算/seed/训练行为，仅修订物理标签与支持性检查。

修订保持全部robot.indexing.geom_ids；named:与body:标签命名空间隔离，空名按父body稳定名称+局部geom序号，不依赖terrain全局ID。reviewer独立CPU MuJoCo检查多个空名geom全部保留、故意重名文本标签无碰撞、插入terrain后digest相同、仅改空名geom摩擦digest改变。native G1 CPU spec gold：30body/68geom（35空名）/35DOF完整覆盖。操作者修订后41相关CPU回归/4语法PASS。

此前7Important修复保持：load后counter归零、原生DOF地址、字符串digest、manager构造前随机流配对、绑定核验前不写完成、运行期磁盘门、惰性NumPy代理；smoke成本finally/Adam下界已修。完整state/LR/RNG、logger/save、wrapper、adapter路由、8family及候选/flat保护仍无重要问题。独立audit源码未变。

本review只读/静态/CPU，无文件依赖修改/环境步进/GPU采样；真实hfield接触/容量/运行行为依旧须预锁smoke，性能与独立审计待正式数据。
