# SRDTrans Temporal Compression Modification Summary

## 修改概述
在 SRDTrans 中添加可配置的 temporal compression 支持，用于 temporal-only ablation 研究。

**核心改动原则**：最小改动，不改变网络整体逻辑，保持参数量一致。

---

## 修改的文件清单

### 1. **train.py** ✅
**行数**: 新增和修改约 5 行

**具体改动**：
- **第 24 行**：添加命令行参数
  ```python
  parser.add_argument('--temporal_strides', type=str, default='2,2,2,2', 
                      help="temporal strides for each encoder/decoder layer, comma-separated (e.g. '2,2,2,2')")
  ```

- **第 209-236 行** (`build_denoise_generator` 函数)：
  - 解析 `--temporal_strides` 字符串为 `list[int]`
  - 传入 `SRDTrans(..., temporal_strides=temporal_strides)`

**验收标准**：
- ✅ 默认值：`"2,2,2,2"`
- ✅ 参数传递给 SRDTrans

---

### 2. **test.py** ✅
**行数**: 新增和修改约 5 行

**具体改动**：
- **第 33 行**：添加相同的命令行参数
- **第 114-127 行** (`build_denoise_generator` 函数)：
  - 同 train.py，解析并传入 `temporal_strides`

**关键要求**：
- ✅ test.py 必须与 train.py 使用完全相同的参数
- ✅ 否则 checkpoint 加载会失败

---

### 3. **SRDTrans/MainFrame.py** ✅
**行数**: 新增和修改约 30 行

#### 3a. MainFrame.__init__() 改动（第 7-38 行）

**新增参数**：`temporal_strides=None`

**新增逻辑**：
```python
# 验证和设置 temporal_strides
if temporal_strides is None:
    temporal_strides = [2] * len(f_maps)
assert len(temporal_strides) == len(f_maps), \
    f"temporal_strides length {len(temporal_strides)} != f_maps length {len(f_maps)}"
for stride in temporal_strides:
    assert stride in [1, 2], f"temporal_stride must be 1 or 2, got {stride}"
self.temporal_strides = temporal_strides
```

**调用改动**：
- `self.encoders = self.temporalSqueeze(..., temporal_strides=temporal_strides)`
- `self.decoders = self.temporalExcitation(..., temporal_strides=temporal_strides[::-1])`

#### 3b. SqueezeLayer 改动（第 73-92 行）

**新增参数**：`temporal_stride=2`

**改动行**：第 87 行
```python
# 原始：stride=(2,1,1)
# 改为：stride=(temporal_stride,1,1)
self.down_sample = nn.Conv3d(out_channels, out_channels, kernel_size=(3,3,3), 
                             stride=(temporal_stride,1,1), padding=(1,1,1))
```

#### 3c. ExcitationLayer 改动（第 95-119 行）

**新增参数**：`temporal_stride=2`

**改动行**：第 111 行
```python
# 原始：stride=(2,1,1)
# 改为：stride=(temporal_stride,1,1)
self.up_sample = nn.ConvTranspose3d(in_channels=in_channels, out_channels=in_channels, 
                                    kernel_size=(4,3,3), stride=(temporal_stride,1,1), padding=(1,1,1))
```

#### 3d. temporalSqueeze() 改动（第 36-48 行）

**新增参数**：`temporal_strides`

**改动**：传入每层对应的 stride
```python
encoder_layer = SqueezeLayer(
    in_channels=f_maps[idx-1],
    out_channels=f_maps[idx],
    temporal_stride=temporal_strides[idx-1]  # 根据层级选择 stride
)
```

#### 3e. temporalExcitation() 改动（第 50-62 行）

**新增参数**：`temporal_strides`

**改动**：decoder 的 stride 反向对应 encoder
```python
decoder_layer = ExcitationLayer(
    in_channels=f_maps[idx-1],
    out_channels=f_maps[idx],
    if_up_sample=True,
    temporal_stride=temporal_strides[idx-1]  # 已反向，正确对应
)
```

#### 3f. forward() 改动（第 70-88 行）

**新增逻辑**：形状检查
```python
def forward(self, x):
    input_shape = x.shape
    # ... 处理 ...
    # 检查输出形状与输入一致
    assert x.shape == input_shape, \
        f"Output shape {x.shape} doesn't match input shape {input_shape}"
    return x
```

**目的**：确保 encoder-decoder 对齐，检测 padding 导致的形状不匹配

---

### 4. **SRDTrans/__init__.py** ✅
**行数**: 新增和修改约 25 行

#### 4a. 导入改动（第 1-6 行）
```python
from functools import reduce
import operator
```
用于计算 `temporal_strides` 的乘积。

#### 4b. SRDTrans.__init__() 改动（第 9-70 行）

**新增参数**：`temporal_strides=None`

**新增验证逻辑**：
```python
# Process temporal_strides
if temporal_strides is None:
    temporal_strides = [2] * len(f_maps)

# Validate temporal_strides
assert isinstance(temporal_strides, (list, tuple)), "temporal_strides must be a list or tuple"
assert len(temporal_strides) == len(f_maps), \
    f"len(temporal_strides)={len(temporal_strides)} != len(f_maps)={len(f_maps)}"

# Calculate total compression and seq_length
total_compression = reduce(operator.mul, temporal_strides, 1)

for stride in temporal_strides:
    assert stride in [1, 2], f"temporal_stride must be 1 or 2, got {stride}"

assert img_time % total_compression == 0, \
    f"img_time ({img_time}) must be divisible by product of temporal_strides ({total_compression})"

seq_length = img_time // total_compression
```

**关键改动**：第 54 行
```python
# 原始：seq_length=img_time//(2**len(f_maps))  # 固定 16x 压缩
# 改为：seq_length = img_time // total_compression  # 可配置压缩
```

**日志输出**：
```python
print(f"[SRDTrans] patch_t={img_time}, temporal_strides={temporal_strides}, "
      f"temporal_compression={total_compression}x, T_core={seq_length}")
```

**参数传递**：
```python
super(SRDTrans, self).__init__(
    img_dim, img_time, in_channel,
    f_maps=f_maps,
    input_dropout_rate=input_dropout_rate,
    temporal_strides=temporal_strides  # 传入 MainFrame
)
```

---

## 实验配置示例

### Group A: 固定 T_core=8，测试 total temporal compression
```bash
# E01: 1x compression
python train.py --patch_t 8 --temporal_strides "1,1,1,1"

# E02: 2x compression
python train.py --patch_t 16 --temporal_strides "2,1,1,1"

# E03: 4x compression
python train.py --patch_t 32 --temporal_strides "2,2,1,1"

# E04: 8x compression
python train.py --patch_t 64 --temporal_strides "2,2,2,1"

# E05: 16x compression（原始配置）
python train.py --patch_t 128 --temporal_strides "2,2,2,2"
```

### Group B: Early vs Late compression
```bash
# E06: 后压缩 2x
python train.py --patch_t 16 --temporal_strides "1,1,1,2"

# E07: 后压缩 4x
python train.py --patch_t 32 --temporal_strides "1,1,2,2"

# E08: 后压缩 8x
python train.py --patch_t 64 --temporal_strides "1,2,2,2"
```

### Group C: 更大 T_core=16
```bash
# E09: 1x, T_core=16
python train.py --patch_t 16 --temporal_strides "1,1,1,1"

# E10: 2x, T_core=16
python train.py --patch_t 32 --temporal_strides "2,1,1,1"

# E11: 4x, T_core=16
python train.py --patch_t 64 --temporal_strides "2,2,1,1"

# E12: 8x, T_core=16
python train.py --patch_t 128 --temporal_strides "2,2,2,1"
```

---

## 验收标准检查清单

### ✅ 参数流向
- [ ] train.py: 命令行参数 → args.temporal_strides
- [ ] test.py: 命令行参数 → args.temporal_strides  
- [ ] build_denoise_generator: args.temporal_strides → SRDTrans
- [ ] SRDTrans.__init__: 验证并计算 seq_length
- [ ] MainFrame.__init__: 接收并验证 temporal_strides
- [ ] SqueezeLayer/ExcitationLayer: 使用对应的 stride

### ✅ 功能正确性
- [ ] `temporal_strides` 默认值为 `[2,2,2,2]`
- [ ] 每个 stride 只能是 1 或 2
- [ ] `len(temporal_strides) == len(f_maps)` （必须 4 个值对应 f_maps=[8,16,32,64]）
- [ ] `img_time % product(temporal_strides) == 0` （整除检查）
- [ ] `seq_length = img_time // product(temporal_strides)` （正确计算）
- [ ] Decoder stride 正确反向对应 encoder

### ✅ 网络结构
- [ ] 不增加新的模块
- [ ] 不改变 STB 主体
- [ ] 不删除任何层
- [ ] 使用 ConvTranspose3d (stride=1) 而非 Identity（保持参数量）
- [ ] 不改变 spatial compression（H, W 维度 stride 保持 (1,1)）
- [ ] 输出形状与输入形状一致

### ✅ 向后兼容性
- [ ] 不指定 `--temporal_strides` 时，自动使用默认 `"2,2,2,2"`（等同原始行为）
- [ ] train.py 和 test.py 必须使用相同的 `temporal_strides`
- [ ] Checkpoint 加载正常（同步验证）

### ✅ 日志输出
- [ ] 训练开始时打印：patch_t, temporal_strides, temporal_compression, T_core
- [ ] 例：`[SRDTrans] patch_t=64, temporal_strides=[2, 2, 2, 1], temporal_compression=8x, T_core=8`

---

## 测试方法

### 快速验证（不训练）
```bash
cd /data/zhouxirou/Ours_core/prior/srdtrans

# 测试默认配置
python -c "
from SRDTrans import SRDTrans
import torch

model = SRDTrans(
    img_dim=128, img_time=64, in_channel=1,
    embedding_dim=128, num_heads=8, hidden_dim=512,
    window_size=7, num_transBlock=1, attn_dropout_rate=0.1,
    f_maps=[8,16,32,64], temporal_strides=[2,2,2,1]
)

x = torch.randn(1, 1, 64, 128, 128)
y = model(x)
print(f'Input shape: {x.shape}, Output shape: {y.shape}')
assert y.shape == x.shape, 'Shape mismatch!'
print('✅ Shape check passed!')
"

# 测试不同配置
python -c "
from SRDTrans import SRDTrans
import torch

configs = [
    ([1,1,1,1], 8, 8),    # 1x, T_core=8
    ([2,1,1,1], 16, 8),   # 2x, T_core=8
    ([2,2,1,1], 32, 8),   # 4x, T_core=8
]

for strides, patch_t, expected_core in configs:
    model = SRDTrans(
        img_dim=128, img_time=patch_t, in_channel=1,
        embedding_dim=128, num_heads=8, hidden_dim=512,
        window_size=7, num_transBlock=1, attn_dropout_rate=0.1,
        f_maps=[8,16,32,64], temporal_strides=strides
    )
    print(f'✅ Config {strides} created successfully (patch_t={patch_t})')
"
```

### 完整训练测试
```bash
# 测试 E04 配置
python train.py \
    --patch_t 64 \
    --temporal_strides "2,2,2,1" \
    --n_epochs 1 \
    --datasets_folder ./datasets/sample \
    --GPU 0

# 验证 test.py
python test.py \
    --patch_t 64 \
    --temporal_strides "2,2,2,1" \
    --model_input_path ./pth/model.pth \
    --test_data_dir ./datasets/test
```

---

## 注意事项

1. **checkpoint 兼容性**：
   - 旧的 checkpoint（使用默认 `[2,2,2,2]`）可以直接加载
   - 如果要加载为其他配置，需要确保 model 的 `f_maps` 不变

2. **参数数量稳定性**：
   - 当 stride=1 时，ExcitationLayer 的 ConvTranspose3d 使用 `stride=(1,1,1)` 
   - 保持了参数量与原始 stride=2 时基本相同
   - 不使用 `nn.Identity()` 因为会改变参数量

3. **形状检查**：
   - 如果输出形状与输入不一致，会抛出 AssertionError
   - 这通常表示 padding 配置不合理（需要检查 kernel_size 和 stride 配置）

4. **测试数据准备**：
   - 确保 `img_time` 能被 `product(temporal_strides)` 整除
   - 例如 patch_t=64 和 strides=[2,2,2,1] → 64 % 8 == 0 ✅

---

## 改动统计

| 文件 | 新增行 | 修改行 | 删除行 | 总计 |
|-----|-------|-------|-------|------|
| train.py | 3 | 2 | 0 | 5 |
| test.py | 3 | 2 | 0 | 5 |
| MainFrame.py | 15 | 15 | 0 | 30 |
| SRDTrans/__init__.py | 22 | 3 | 0 | 25 |
| **总计** | **43** | **22** | **0** | **65** |

✅ **最小改动完成** - 只修改必要部分，不影响网络逻辑
