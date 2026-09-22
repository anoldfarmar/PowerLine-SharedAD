# 任务完成后的远程同步

本文件补充 CLAUDE.md，不替代任务合同、冻结区和人工门规则。

每个已授权任务在完成验收和 log_run 登记后、向人类报告前，执行一次 Git 同步：

1. 查看 git status、当前分支及 remote；只允许在预期实验仓库和预期分支操作。
2. 仅暂存本任务新增或修改的代码、报告、机器可读指标及 registry/runs.jsonl。
   逐个指定路径；禁止 git add -A、git add .。
3. 检查 git diff --cached --name-only 和 git diff --cached --stat。
   不得提交原始数据、密钥、凭证、模型大权重、缓存、临时日志，
   也不得提交未经人类授权修改的 plan/**、registry/gates/**、data/splits/**。
4. 验收 PASS 时提交本任务结果；验收 FAIL 时只提交允许登记的失败诊断，
   不得把失败标为 PASS。
5. 使用有意义的提交信息，并执行 git push origin 当前分支。
   禁止 force push、改写历史、自动解决冲突或自行创建人工门标志。
6. push 成功后核对本地 HEAD 与 origin/当前分支一致，
   再向人类报告 commit SHA、远程分支、验收状态。
7. 若认证失败、远程领先、存在冲突或发现疑似敏感文件：
   停止推送，保留本地改动，报告确切原因；不得强推或静默声称已同步。