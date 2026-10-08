# USFM 单帧分割基线

本项目采用 [openmedlab/USFM](https://github.com/openmedlab/USFM) 的预训练超声图像主干，加官方 SegVit 分割头，在本项目的 TEE 标注帧上做有监督微调。这里先建立**逐帧分割基线**；时序模块应在该基线之上另做同主干消融。

## 已有文件

- `external/USFM/`：官方源码，本机已克隆；当前检出的提交为 `960a1e1d30b9490e53e6e7cf1b4dd24425fffc67`，该目录被 `.gitignore` 忽略。
- `configs/usfm_tee_valve.yaml`：官方 `SegBaseDataset` 数据配置。官方 README 中写 `trainning_set`，但其 `configs/data/Seg/toy_dataset.yaml` 和代码实际读取 `training_set`；转换器采用后者。
- `code/scripts/prepare_usfm_dataset.py`：原始 NIfTI 的每帧转换为 PNG，生成 `training_set/val_set/test_set` 的 `image/` 和 `mask/`，并保存 `frames.csv` 回溯映射。图像保持原始 `uint8` 强度，非 `uint8` 沿用本项目的每序列分位数映射；标签输出 0/255，官方二值读取后得到 0/1。
- `code/scripts/run_usfm.py`：检查数据、权重和官方源码，复制项目配置到官方源码，启动官方 SegVit 训练或测试。默认只打印命令，加入 `--execute` 才运行。
- `code/scripts/eval_usfm_predictions.py`：将官方测试输出的 PNG 掩膜以最近邻映射回原始 NIfTI 像素网格，按帧、序列和 SAX/LAX 分别计算 Dice/IoU。
- `code/scripts/export_usfm_prediction_gifs.py`：将官方测试预测 `mask_pre/` 按 `frames.csv` 还原为序列 GIF，三栏显示原图、真值和 USFM 预测。
- `code/scripts/make_usfm_overfit.py`：复制一张有前景的训练帧到诊断集的 train/val/test，检查模型能否记住这一帧；这个诊断集不能用于泛化评估。
- `code/scripts/train_usfm_partial.py`：在官方 SegVit 训练器上施加分层冻结与双学习率；由 `run_usfm.py --freeze-first-blocks` 调用。

## USFM 前层冻结、后层与分割头训练

本项目的 USFM 主干有 12 个 ViT block。以下方案冻结 `patch_embed`、`cls_token` 和 `blocks.0–7`，训练 `blocks.8–11`、FPN 与 SegVit decoder。FPN 在官方预训练权重中没有对应参数，因此必须训练。两组实际学习率分别为 `1e-5` 和 `1e-4`；这里直接设置 AdamW 参数组，不经过上游按 batch size/512 的统一学习率缩放。

先用已有 pilot 数据检查训练曲线：

```powershell
$env:MPLCONFIGDIR = (Resolve-Path outputs).Path
$env:NO_ALBUMENTATIONS_UPDATE = '1'
& '.venv-usfm\Scripts\python.exe' code/scripts/run_usfm.py --dataset outputs/usfm_pilot --output outputs/usfm_partial_pilot_8 --freeze-first-blocks 8 --backbone-lr 0.00001 --head-lr 0.0001 --batch-size 1 --epochs 40 --val-freq 5 --execute
```

之后在完整探索性训练集上训练：

```powershell
& '.venv-usfm\Scripts\python.exe' code/scripts/run_usfm.py --dataset outputs/usfm_tee_valve --output outputs/usfm_partial_full_8 --freeze-first-blocks 8 --backbone-lr 0.00001 --head-lr 0.0001 --batch-size 1 --epochs 40 --val-freq 5 --execute
```

`--execute` 才启动训练；去掉它可先检查将执行的命令。训练开始时日志显示冻结/可训练参数量和两组学习率；每个 epoch 从第 1 个 batch 开始打印进度，之后默认每 **50** 个 batch 打印一次 loss、前景 Dice、学习率和预计剩余时间，验证阶段也会打印。需要更频繁时加 `--log-every 10`。屏幕和 `outputs/usfm_partial_full_8/outputs/log_rank0.log` 都能看到相同信息。这个入口强制关闭梯度检查点、使用 `accumulation_steps=1` 与 `warmup_epochs=0`，因为冻结后的检查点反传和上游累积梯度实现存在风险。最佳权重保存在运行目录的 `outputs/best*.pth`。完整数据需先核实患者级划分，当前结果仍属探索性实验。

完整训练完成后，用新权重测试并生成 GIF；测试入口只加载模型参数，因此可读取采用不同优化器参数组保存的权重：

```powershell
$best = Get-ChildItem 'outputs/usfm_partial_full_8/outputs/best*.pth' | Select-Object -First 1
if (-not $best) { throw '没有找到最佳分割权重，请检查训练日志' }
& '.venv-usfm\Scripts\python.exe' code/scripts/run_usfm.py --mode test --dataset outputs/usfm_tee_valve --resume $best.FullName --output outputs/usfm_partial_full_8_test --img-size 224 --batch-size 1 --execute
if ($LASTEXITCODE -ne 0) { throw 'USFM 测试失败，请检查上方报错' }
$pred = Get-ChildItem 'outputs/usfm_partial_full_8_test/outputs' -Directory -Filter 'best_test_dice*' | Select-Object -First 1
if (-not $pred) { throw '未找到测试预测目录' }
& '.venv\Scripts\python.exe' code/scripts/export_usfm_prediction_gifs.py --dataset outputs/usfm_tee_valve --predictions (Join-Path $pred.FullName 'mask_pre') --output outputs/usfm_partial_full_8_test/gifs --fps 8 --stride 1
```

测试时不要传 `--freeze-first-blocks`；该参数只影响训练。`best*.pth` 来自验证集选模，测试集不参与训练。

## SAX 与 LAX 分开训练

SAX 与 LAX 的前景面积和外观差异很大，可以把分开训练作为与上面的联合模型并列的对照。现有 `prepare_usfm_dataset.py --views` 已支持按切面导出；相同种子下，它沿用联合导出的每个切面内的序列划分。以下命令需在项目根目录运行；同名输出目录会自动清空并重建。SAX 有一例图像/标签仿射冲突，因此探索性导出使用 `--skip-invalid`。

```powershell
& '.venv\Scripts\python.exe' code/scripts/prepare_usfm_dataset.py --views SAX --output outputs/usfm_sax_only --skip-invalid
if ($LASTEXITCODE -ne 0) { throw 'SAX 数据导出失败' }
& '.venv\Scripts\python.exe' code/scripts/prepare_usfm_dataset.py --views LAX --output outputs/usfm_lax_only
if ($LASTEXITCODE -ne 0) { throw 'LAX 数据导出失败' }
if (-not (Test-Path outputs/usfm_sax_only/export.json) -or -not (Test-Path outputs/usfm_lax_only/export.json)) { throw '导出未完成，检查 _INCOMPLETE 和上面的错误信息' }
$env:MPLCONFIGDIR = (Resolve-Path outputs).Path
$env:NO_ALBUMENTATIONS_UPDATE = '1'
& '.venv-usfm\Scripts\python.exe' code/scripts/run_usfm.py --dataset outputs/usfm_sax_only --output outputs/usfm_sax_partial_8 --freeze-first-blocks 8 --backbone-lr 0.00001 --head-lr 0.0001 --batch-size 1 --epochs 40 --val-freq 5 --log-every 50 --execute
```

**等 SAX 训练结束后**，再在同一张 GPU 上启动 LAX：

```powershell
& '.venv-usfm\Scripts\python.exe' code/scripts/run_usfm.py --dataset outputs/usfm_lax_only --output outputs/usfm_lax_partial_8 --freeze-first-blocks 8 --backbone-lr 0.00001 --head-lr 0.0001 --batch-size 1 --epochs 40 --val-freq 5 --log-every 50 --execute
```

各自用其验证集选 `best*.pth`，再只在对应切面的测试集评估。示例：

```powershell
$bestSax = Get-ChildItem 'outputs/usfm_sax_partial_8/outputs/best*.pth' | Select-Object -First 1
if (-not $bestSax) { throw '未找到 SAX 最佳权重' }
& '.venv-usfm\Scripts\python.exe' code/scripts/run_usfm.py --mode test --dataset outputs/usfm_sax_only --resume $bestSax.FullName --output outputs/usfm_sax_partial_8_test --batch-size 1 --execute
$bestLax = Get-ChildItem 'outputs/usfm_lax_partial_8/outputs/best*.pth' | Select-Object -First 1
if (-not $bestLax) { throw '未找到 LAX 最佳权重' }
& '.venv-usfm\Scripts\python.exe' code/scripts/run_usfm.py --mode test --dataset outputs/usfm_lax_only --resume $bestLax.FullName --output outputs/usfm_lax_partial_8_test --batch-size 1 --execute
```

比较时在**同一切面、同一测试病例**上对照联合模型与切面专用模型，分别报告 SAX/LAX 的逐序列指标。分开训练每个模型见到的病例更少，因此效果是否更好要由验证与测试结果决定；当前文件夹划分仍未证明患者级独立。原有联合训练目录和结果保留，不需要改写。

## 直接预测现有完整测试集并导出 GIF

在项目根目录的 **PowerShell** 中逐行运行下面命令。这里使用已有的 `best8.pth` 分割权重预测 `outputs/usfm_tee_valve` 的完整测试集（SAX 5 条、LAX 4 条，共 879 帧）。模型输入只有单帧图像；测试标签仅供评分和 GIF 对照，不作为模型提示。

```powershell
$env:MPLCONFIGDIR = (Resolve-Path outputs).Path
$env:NO_ALBUMENTATIONS_UPDATE = '1'
& '.venv-usfm\Scripts\python.exe' code/scripts/run_usfm.py --mode test --dataset outputs/usfm_tee_valve --resume outputs/usfm_pilot_train_224b/outputs/best8.pth --output outputs/usfm_full_test_from_pilot --img-size 224 --batch-size 1 --execute
if ($LASTEXITCODE -ne 0) { throw 'USFM 测试失败，请检查上方报错' }
$pred = Get-ChildItem 'outputs/usfm_full_test_from_pilot/outputs' -Directory -Filter 'best_test_dice*' | Select-Object -First 1
if (-not $pred) { throw '未找到测试预测目录，请检查 USFM 测试日志' }
& '.venv\Scripts\python.exe' code/scripts/export_usfm_prediction_gifs.py --dataset outputs/usfm_tee_valve --predictions (Join-Path $pred.FullName 'mask_pre') --output outputs/usfm_full_test_from_pilot/gifs --fps 8 --stride 1
if ($LASTEXITCODE -ne 0) { throw 'GIF 导出失败，请检查上方报错' }
& '.venv\Scripts\python.exe' code/scripts/eval_usfm_predictions.py --dataset outputs/usfm_tee_valve --predictions (Join-Path $pred.FullName 'mask_pre') --output outputs/usfm_full_test_from_pilot/scores
if ($LASTEXITCODE -ne 0) { throw '测试评分失败，请检查上方报错' }
Get-Content outputs/usfm_full_test_from_pilot/scores/summary.json
```

预测 PNG 位于 `outputs/usfm_full_test_from_pilot/outputs/best_test_dice*/mask_pre/`；每条测试序列的动图位于 `outputs/usfm_full_test_from_pilot/gifs/`。GIF 三栏依次为超声图、绿色真值、红色预测，顶部是帧序号与该帧 Dice。GIF 播放的 8 fps 只控制观看速度，并非采集帧率。改变 `--stride 2` 可减小 GIF 文件；输出目录已存在时请换新名字，脚本不会覆盖旧结果。

当前 USFM 模型逐帧计算 `ŷₜ=f(xₜ)`，既不读取前帧，也不读取后帧。GIF 只是把已独立预测的帧按 NIfTI 第三轴索引升序排列；现有导出没有原始采集时间戳，**索引升序是否等于真实时间正向尚未核实**。后续若做在线时序模型，必须先核实方向，并限制第 `t` 帧预测仅使用 `x₀…xₜ` 和允许的历史状态，不能使用 `xₜ₊₁…` 或未来标签。

**这个现成分割权重只用 8 条序列、每条 16 帧完成探索性训练，已有小样本验证 Dice 约 0.077。** 上述命令能检查完整测试集的预测和可视化流程，但不能把结果当成完整训练后的 USFM 性能。官方 `USFM_latest.pth` 只有预训练主干，没有可直接用于瓣膜分割的现成分割头；如需可靠的单帧基线，应按下文用训练集训练分割头并在验证集选权重，再用同样命令预测测试集。当前划分尚未核实患者级独立性，因此这些分数仍为探索性结果。

## 一帧过拟合诊断

按当前已导出的原始尺寸 PNG 掩膜统计，训练集 LAX 前景像素占 **0.396%**（3,736 帧，其中 31 帧为空），SAX 占 **3.766%**（4,545 帧，其中 238 帧为空）。这是所有像素汇总后的比例；模型在 224×224 上训练，缩放后个别极小目标仍应单独核查。

先检查训练通路能否记住**同一张训练图像和掩膜**。下面故意把一张 LAX 训练帧复制到 train/val；官方加载器还要求一个 test 目录，因此其中也放同一帧占位。这个数据集存在刻意的同帧重叠，只能做记忆检查；验证 Dice 不是独立验证或预测性能，脚本会拒绝对它执行 `--mode test`。诊断集只含训练帧，不读取原测试集，也不检验时序因果性。

```powershell
& '.venv\Scripts\python.exe' code/scripts/make_usfm_overfit.py --source outputs/usfm_tee_valve --output outputs/usfm_overfit_lax_one --view LAX
$env:MPLCONFIGDIR = (Resolve-Path outputs).Path
$env:NO_ALBUMENTATIONS_UPDATE = '1'
& '.venv-usfm\Scripts\python.exe' code/scripts/run_usfm.py --dataset outputs/usfm_overfit_lax_one --output outputs/usfm_overfit_lax_run --epochs 200 --batch-size 1 --effective-lr 0.0001 --warmup-epochs 0 --val-freq 10 --execute
Select-String -Path outputs/usfm_overfit_lax_run/outputs/log_rank0.log -Pattern 'Train:|Dice of the network|Current learning rate' | Select-Object -Last 30
```

若同一帧的验证 Dice 接近 1，说明模型、标签转换、输出和优化器至少能完成基本记忆；若一直接近 0，先排查学习率、掩膜、输出和损失，暂不要把完整测试集低分解释为 USFM 表征能力不足。当前官方训练变换仍含同步旋转/翻转；因此这个实验检验的是同一帧及其增强版本，失败时还需要排查增强。`--effective-lr` 是官方按 batch size 和梯度累积数缩放后的目标峰值学习率；原有小样本试跑的配置为 10 轮训练、20 轮 warmup、batch 1、`accumulation_steps=4`，日志中的学习率约为 4×10⁻⁷ 到 1.3×10⁻⁶。上游代码每批都 `optimizer.step()`，其 `accumulation_steps` 当前只影响学习率缩放，未真正累积梯度。

## 1. 准备数据

先在项目根目录运行审计和转换。以下 PowerShell 命令使用现有 `.venv`：

```powershell
& '.venv\Scripts\python.exe' code/scripts/audit_dataset.py --output outputs/data_audit_usfm
& '.venv\Scripts\python.exe' code/scripts/prepare_usfm_dataset.py --output outputs/usfm_tee_valve --skip-invalid
```

当前没有已核实的患者身份映射时，转换器只从已有训练文件夹按**完整序列**抽验证集，保留已有测试文件夹。`export.json` 标记为 `exploratory_folder_split`。SAX 训练集第 27 例图像/标签仿射不一致，普通导出会拒绝它；上面的 `--skip-invalid` 只用于探索性导出，并将排除原因写入 `export.json`。这适合调通模型，**不能证明患者级独立，也不能作为论文最终测试划分**。

正式实验先填写 `metadata/patient_manifest.template.csv` 的 `anonymous_patient_id`、`source_image`、`split` 和 `review_status`。每条主集序列均须列出，`source_image` 可以是相对于 `dataset/TEEdataset` 的路径；核查后将 `review_status` 填为 `approved`。同一患者只能出现在一个 split。然后另选一个新输出目录：

```powershell
& '.venv\Scripts\python.exe' code/scripts/prepare_usfm_dataset.py --patient-manifest metadata/patient_manifest.csv --output outputs/usfm_tee_valve_patient
```

转换器会自动清空同名的 `outputs/` 子目录并重建，之前的导出文件会丢失；原始 NIfTI 保持不变。`run_usfm.py --execute` 也会清空同名训练/测试输出目录，包括已有日志与权重。去掉 `--execute` 只打印命令，不清空目录。输出目录若包含当前输入数据、预训练权重或 `--resume` 权重，脚本会拒绝覆盖。当前数据中已有仿射冲突、重复图像与可疑空标签记录，正式训练前应人工处理；参见 [数据与评估协议](DATA_AND_EVALUATION.md)。

## 2. 当前机器的准备状态

以下命令均在 **PowerShell 的项目根目录**运行。项目中已有 Windows 独立环境 `.venv-usfm`、官方源码 `external/USFM`、官方预训练权重 `checkpoints/USFM_latest.pth`，以及分别按切面导出的 `outputs/usfm_sax_only`、`outputs/usfm_lax_only`。文档中的联合数据目录 `outputs/usfm_tee_valve` 如需使用，应先按下文重新导出。权重来源为[官方 USFM 仓库](https://github.com/openmedlab/USFM)，已下载文件的 SHA256 为 `d5fdab3edd140e4ca61471bb4087f91cd7ff2ce270db71b9cab30feda881bd17`，记录见 `checkpoints/USFM_latest.pth.download.json`。`.venv-usfm` 使用 Python 3.11、PyTorch 2.2/cu121 和 MMCV 2.2.0 的 Windows 轮子；这与官方 README 建议的 Linux/Python 3.9/PyTorch 2.4/cu118 组合不同。

**USFM 预训练权重只有图像主干，没有针对你们瓣膜的现成分割头。** 要测试“单帧分割效果”，必须先用训练集的逐帧标签训练分割头，再在独立测试集上推理。训练和测试都使用 `.venv-usfm`；评分脚本使用原有 `.venv` 读取 NIfTI。

目前入口固定为 **224×224**。实测官方权重在 224 下参数、特征和损失均为有限值；官方代码把它插值到 256 时生成 `NaN`，因此当前入口会拒绝其他尺寸。不要将之前 256 尺寸试跑的 0 分当成模型性能。

## 3. 快速流程试跑：少量序列

`outputs/usfm_pilot` 已准备好：SAX/LAX 各有 4 条训练序列、1 条验证序列、1 条测试序列，每条抽取 16 帧。若目录不存在，可自己生成：

```powershell
& '.venv\Scripts\python.exe' code/scripts/make_usfm_pilot.py --source outputs/usfm_tee_valve --output outputs/usfm_pilot
```

### 直接测试已有的小样本分割权重

本机已有一次 224×224 的探索性训练权重 `outputs/usfm_pilot_train_224b/outputs/best8.pth`。若只想先看预测和评分，可直接运行以下命令；这是 **8 条训练序列、每条 16 帧**得到的小样本模型，不能代表完整数据效果：

```powershell
$env:MPLCONFIGDIR = (Resolve-Path outputs).Path
$env:NO_ALBUMENTATIONS_UPDATE = '1'
& '.venv-usfm\Scripts\python.exe' code/scripts/run_usfm.py --mode test --dataset outputs/usfm_pilot --resume outputs/usfm_pilot_train_224b/outputs/best8.pth --output outputs/my_usfm_existing_test --img-size 224 --batch-size 1 --execute
$pred = Get-ChildItem 'outputs/my_usfm_existing_test/outputs' -Directory -Filter 'best_test_dice*' | Select-Object -First 1
if (-not $pred) { throw '没有找到测试掩膜，请检查测试日志' }
& '.venv\Scripts\python.exe' code/scripts/eval_usfm_predictions.py --dataset outputs/usfm_pilot --predictions (Join-Path $pred.FullName 'mask_pre') --output outputs/my_usfm_existing_score
Get-Content outputs/my_usfm_existing_score/summary.json
```

### 自己从预训练主干重新训练

先训练。已有的 `--output` 目录会在 `--execute` 时清空；将下方 `my_usfm_pilot_train` 换成你自己的运行名即可。去掉 `--execute` 时脚本只显示将执行的命令。

```powershell
$env:MPLCONFIGDIR = (Resolve-Path outputs).Path
$env:NO_ALBUMENTATIONS_UPDATE = '1'
& '.venv-usfm\Scripts\python.exe' code/scripts/run_usfm.py --dataset outputs/usfm_pilot --output outputs/my_usfm_pilot_train --img-size 224 --batch-size 1 --epochs 10 --gradient-checkpointing --execute
```

训练会在 `outputs/my_usfm_pilot_train/outputs/` 写 `best<epoch>.pth`。选择验证集最佳权重后，运行独立测试；以下 PowerShell 自动找到该目录中的最佳权重文件：

```powershell
$best = Get-ChildItem 'outputs/my_usfm_pilot_train/outputs/best*.pth' | Select-Object -First 1
if (-not $best) { throw '没有找到最佳分割权重，请先检查训练日志' }
& '.venv-usfm\Scripts\python.exe' code/scripts/run_usfm.py --mode test --dataset outputs/usfm_pilot --resume $best.FullName --output outputs/my_usfm_pilot_test --img-size 224 --batch-size 1 --execute
```

官方测试输出在 `outputs/my_usfm_pilot_test/outputs/best_test_dice*/mask_pre/`。PNG 掩膜像素是 0/1，普通图片查看器中可能几乎全黑；用评分脚本读取，按原始 NIfTI 尺寸计算指标：

```powershell
$pred = Get-ChildItem 'outputs/my_usfm_pilot_test/outputs' -Directory -Filter 'best_test_dice*' | Select-Object -First 1
if (-not $pred) { throw '没有找到测试掩膜，请先检查测试日志' }
& '.venv\Scripts\python.exe' code/scripts/eval_usfm_predictions.py --dataset outputs/usfm_pilot --predictions (Join-Path $pred.FullName 'mask_pre') --output outputs/my_usfm_pilot_score
Get-Content outputs/my_usfm_pilot_score/summary.json
```

`frames.csv` 给出逐帧 Dice/IoU，`cases.csv` 给出逐序列汇总，`summary.json` 分别汇总 SAX/LAX。**这个小样本试跑只能确认流程，不能代表模型在完整数据集上的效果。**

## 4. 探索性全数据实验

把上面三个命令中的 `--dataset` 都改成 `outputs/usfm_tee_valve`，并使用新的输出目录即可。完整导出包含训练 72 条/8,281 帧、验证 13 条/1,401 帧、测试 9 条/879 帧；SAX 训练集第 27 例因图像/标签仿射不一致已记录并排除。完整训练会明显更久，可先设置 `--epochs 5` 检查损失和验证指标，再用新的运行目录做更长训练。

当前 train/test 来源于原有文件夹，尚未核实患者级独立。因此这些分数只能作为探索结果。正式论文实验需要审核 `metadata/patient_manifest.csv`，重新按患者导出数据，然后给训练和测试命令都加上 `--require-patient-split`。
