# Day32 最终代码详细解读

重点文件：day32/python/day32_supply_chain_pipeline.py。本文按调用顺序介绍参数、变量类型、返回值和数据流。

## 一、模块职责与调用关系

| 模块 | 最终流水线调用的功能 |
|---|---|
| day31_ci_cd_pipeline | 阶段定义、版本校验、子进程运行与环境变量 |
| day32_sbom_basic | 制品清单验证、依赖解析、SHA-256、原子JSON写入 |
| day32_dependency_scan | demo_scan和online_scan |
| day32_ed25519_artifact | 读取既有私钥、检查配对公钥、签署声明 |
| day32_signature_gate | 固定公钥指纹和制品验签 |

最终脚本不调用Day31的main，也不调用其发布提升函数。它仅复用五个基础阶段，然后由Day32自己的供应链门禁决定是否提升。

## 二、路径与命令行

PROJECT_ROOT类型是Path，通过脚本绝对路径向上两层得到项目根目录。sys.path加入day31/python，让普通Python导入找到前一天的模块。

两个参数：

- --version：默认32.0.0，通过validate_version限制格式。
- --scan-mode：demo或online，默认demo。

```bash
python day32/python/day32_supply_chain_pipeline.py
python day32/python/day32_supply_chain_pipeline.py --scan-mode online --version 32.0.0
```

demo始终不能正式放行，即使未来演示规则恰好没有风险，也不能把模拟结果当成真实漏洞查询。

## 三、run_id、run_dir与report

run_id是UTC时间加随机后缀，类型str。run_dir是本次流水线独立目录，类型Path。

report是dict，初始状态：

```python
status = "running"
promotion = "skipped"
exit_code = 1
```

默认失败和默认不提升是有意设计：只有完整执行到成功末尾才改为通过。某个中间步骤抛异常时，不会误用默认成功值。

## 四、复用Day31五个阶段

```python
for definition in build_stages(args.version):
    result = run_stage(definition, run_dir, ci_environment(run_id, args.version))
```

数据类型：

- definition：StageDefinition，包含名称、说明、命令元组。
- result：StageResult，包含状态、退出码、耗时、日志路径和错误。
- asdict(result)：转成可JSON序列化的dict。

依次执行质量门禁、HTTP契约、Fake集成、并发SLO、制品构建。

run_stage通过subprocess.run启动独立进程。返回非passed时抛ValueError，控制流直接进入外层except，后面的供应链步骤不执行。

## 五、复制本次制品快照

built是Day31构建目录中对应版本的tar.gz；artifact是当前run_dir下的副本。

后续SBOM、扫描、签名和验签都以artifact为输入，避免直接反复读取其他运行可能更新的dist路径。

快照不是操作系统强制只读对象，所以后面仍要重新计算哈希检测变化。

## 六、build_sbom的内部数据流

```text
artifact: Path
→ read_verified_members()
→ manifest: dict + members: dict[str, bytes]
→ parse_dependencies(requirements内容)
→ dependencies: list[dict]
→ sbom: dict
```

read_verified_members只接受普通文件，限制读取体积，拒绝危险路径和重复路径，逐项核对内部清单中的大小和SHA-256。

expected_hash取自sbom的artifact.sha256，作为后续扫描与发布关联的固定参考。

sbom.json通过write_json_atomically保存：临时文件写完、flush、fsync，然后os.replace。

## 七、依赖扫描与放行公式

results是list[dict]，每项记录包名、版本、状态、来源、风险条目及可选错误信息。

online_scan按包名和精确版本查询OSV，并为尚未完整查完的包保留unknown。

```python
unknown = any(item["status"] == "unknown" for item in results)
findings = any(item["findings"] for item in results)
scan_allowed = args.scan_mode == "online" and not unknown and not findings
```

any表示“至少有一项满足条件”。非空findings列表在布尔判断中为True。

依赖扫描报告还保存artifact_sha256，因此可以知道这次结果对应哪个制品。

如果scan_allowed为False，抛出异常：签名和发布都不会发生。模式、风险、查询未知均会阻断。

## 八、签名前复核哈希

```python
if sha256_file(artifact) != expected_hash:
    raise ValueError(...)
```

如果从生成SBOM到扫描结束期间制品被修改，就不能用之前的扫描结果给新内容签名。

这里的保护适用于本地学习流程；真实并发构建环境还应采用只读、按内容寻址的制品存储和隔离签名工作区，减少检查与使用之间的变化窗口。

## 九、私钥加载与签名

```python
private_key = load_private_key(create=False)
ensure_public_key(private_key)
```

private_key是Ed25519PrivateKey对象。create=False要求第4步已完成私钥初始化，不在CI中默默生成新发布身份。

ensure_public_key用于检查已有公私钥配对；如果公钥文件不存在，函数可以导出对应公钥，但不能据此改写固定信任策略。

sign_artifact返回dict：

```text
statement
  schema_version
  algorithm
  artifact_name
  size_bytes
  sha256
  public_key_sha256
signature_base64
```

私钥签的是带域前缀的规范化声明。文件通过声明中的SHA-256与签名绑定。

HMAC练习没有进入最终发布流水线，因为这里使用公钥验签机制。

## 十、evaluate_gate验签门禁

参数依次是：制品路径、签名路径、可信公钥路径、信任策略路径，全部为Path。

返回dict，包含checks列表、decision和exit_code。

检查顺序：

1. 读取独立策略并校验格式。
2. 计算公钥指纹，与策略固定值比较。
3. 用该公钥验证声明签名，再核对实际制品。
4. 验证tar.gz内部清单。

首次失败后停止，后续项标记skipped。流水线仅在decision等于allow时调用提升函数。

公钥验证与私钥签名角色不同：单独验签函数不需要私钥。最终流水线因为还承担签名职责，所以整体运行环境拥有私钥。

## 十一、promote_bundle参数

```python
promote_bundle(version: str, files: list[Path], expected_hash: str) -> Path
```

files按约定顺序传递：

```text
files[0] = 制品
files[1] = Ed25519签名JSON
files[2] = SBOM
files[3] = 依赖扫描报告
files[4] = 验签门禁报告
```

顺序被函数内部使用，调换前两项会导致错误。更大项目可以改成具名dataclass减少位置参数依赖。

## 十二、promote_bundle执行过程

首先检查day32/releases/version是否存在。当前Day32策略是已有版本直接拒绝，不沿用Day31“相同哈希幂等跳过”的行为。

随后在releases目录里创建临时目录，逐文件复制并比较复制前后SHA-256。

再检查制品副本哈希是否仍等于expected_hash，并对副本重新运行evaluate_gate。

最后bundle.rename(target)把完整目录提升到目标版本路径。临时目录位于同一文件系统，目录重命名让读者看到完整发布包。

TemporaryDirectory退出时清理演练临时路径；已移动到正式版本路径的bundle继续保留。

## 十三、成功状态和异常传播

成功末尾：

```python
report.update(status="passed", promotion="promoted", release_directory=str(target))
exit_code = 0
```

任何普通异常进入except，report改为failed，保存异常类型与说明。promotion保持skipped。

finally会尝试保存本次pipeline-report.json和最新汇总报告。磁盘故障等导致报告写入本身失败时，不能保证报告仍能落盘，程序会以异常结束。

基础阶段失败和扫描拒绝有明确阶段记录；签名等操作如果在阶段记录追加前抛异常，具体原因体现在顶层error中。

## 十四、完整数据类型表

| 变量 | 类型 | 用途 |
|---|---|---|
| args | argparse.Namespace | 版本和模式 |
| run_id | str | 本次运行编号 |
| run_dir | Path | 隔离日志与制品目录 |
| result | StageResult | 子进程阶段结果 |
| artifact | Path | 本次待发布快照 |
| sbom | dict | 应用、制品、依赖、文件清单 |
| expected_hash | str | 扫描时制品SHA-256 |
| results | list[dict] | 逐依赖扫描结果 |
| unknown/findings/scan_allowed | bool | 发布条件判断 |
| private_key | Ed25519PrivateKey | 签署制品声明 |
| gate | dict | 验签门禁结果 |
| target | Path | 成功提升后的版本目录 |
| report | dict | 整体过程记录 |
| exit_code | int | 外层CI机器判定 |

## 十五、复习一次默认运行

```text
加载参数 version=32.0.0, scan-mode=demo
→ 创建run_id和run_dir
→ Day31五个阶段通过
→ 复制tar.gz
→ SBOM和内部文件检查通过
→ demo产生虚构风险和unknown
→ scan_allowed=False
→ 抛异常并保存failed报告
→ 签名未执行
→ 发布提升skipped
→ 退出码1
```

这条阻断路径已实际运行。在线无风险时的完整放行路径尚未实际运行，不能把默认演示失败报告当成真实依赖扫描结果。

## 十六、当前产物与后续增强边界

当前对制品本体签署声明，没有单独认证SBOM和扫描报告。报告虽然与制品一起分发，但外部接收者若要验证报告来源，需要进一步为这些证据签名或生成可验证证明。

真实生产环境还需要完整传递依赖SBOM、包分发哈希、漏洞规则策略、密钥轮换、可信策略管理和隔离签名环境。

这些是能力边界，今天不继续增加步骤。Day32的目标是理解从“文件一致”到“可信身份签署”，再到“机器强制执行发布规则”的完整关系。
