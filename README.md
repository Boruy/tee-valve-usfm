# 主动脉瓣 TEE 分割项目

本项目使用超声基础模型 USFM 和 SegVit 分割头，在经食道超声（TEE）主动脉瓣数据上建立**单帧分割基线**。后续研究在相同数据划分和主干上验证时序信息的收益。当前训练和测试划分尚未核实患者级独立性，因此现有指标属于探索性结果。

## 项目目录

```text
code/scripts/          数据审计、USFM 数据准备、训练、测试、评分与 GIF 导出
code/valve_seg/        NIfTI 数据读取、图像转换与分割指标
code/tests/            数据协议和导出逻辑测试
configs/               USFM 数据配置
dataset/TEEdataset/    原始 NIfTI 数据；不修改
docs/                  数据协议与 USFM 操作说明
experiments/           实验记录
idea/                  原版与后续研究构想
metadata/              患者级划分模板
external/USFM/         USFM 官方源码
checkpoints/           USFM 预训练权重
outputs/               数据导出、训练、预测与评分结果
```

## 运行入口

以下命令在项目根目录的 PowerShell 中执行。`.venv` 用于数据处理，`.venv-usfm` 用于训练和测试。完整参数与实验限制见 [USFM 运行说明](docs/USFM.md) 和 [数据与评估协议](docs/DATA_AND_EVALUATION.md)。

```powershell
# 审计原始数据
& '.venv\Scripts\python.exe' code/scripts/audit_dataset.py --output outputs/data_audit

# 按 SAX/LAX 分别导出；同名输出目录会自动重建
& '.venv\Scripts\python.exe' code/scripts/prepare_usfm_dataset.py --views SAX --output outputs/usfm_sax_only --skip-invalid
if ($LASTEXITCODE -ne 0) { throw 'SAX 导出失败' }
& '.venv\Scripts\python.exe' code/scripts/prepare_usfm_dataset.py --views LAX --output outputs/usfm_lax_only
if ($LASTEXITCODE -ne 0) { throw 'LAX 导出失败' }

# SAX 部分解冻训练；--execute 才实际运行
& '.venv-usfm\Scripts\python.exe' code/scripts/run_usfm.py --dataset outputs/usfm_sax_only --output outputs/usfm_sax_partial_8 --freeze-first-blocks 8 --backbone-lr 0.00001 --head-lr 0.0001 --batch-size 1 --epochs 40 --val-freq 5 --execute

# 自动选择最佳权重，测试 SAX 并生成评分和 GIF
& .\code\scripts\test_usfm_views.ps1 -View SAX
```

LAX 训练命令见 [USFM 运行说明](docs/USFM.md#sax-与-lax-分开训练)。两种切面都训练完后，也可以用 `-View Both` 依次测试。测试入口会重新生成同名测试输出目录。

当前探索性导出中的 SAX 异常仿射病例由 `--skip-invalid` 记录并排除。正式实验需要先核实标注质量、帧顺序和患者身份，填写并批准 `metadata/patient_manifest.template.csv` 所示的患者级划分。

## GitHub 仓库说明

仓库只保存项目代码、配置、文档和文本研究构想。原始医学影像、患者级正式清单、预训练权重、第三方源码、实验输出、本地虚拟环境、本地论文 PDF 和未审查的演示文稿均由 `.gitignore` 排除。克隆后需自行准备 `dataset/TEEdataset/`、`external/USFM/` 和 `checkpoints/USFM_latest.pth`，再按 [USFM 运行说明](docs/USFM.md) 导出数据与运行实验。
