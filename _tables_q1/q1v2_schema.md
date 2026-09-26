| 项 | 值 | 说明 |
|---|---|---|
| schema_version | q1v2-feature-v1.0 | NPZ 内嵌，防与旧版产物混淆 |
| NPZ 字段数 | 85 | 含标量、掩码、特征、原生序列、CSR 溯源五类 |
| NPZ 读取方式 | numpy.load(..., allow_pickle=False) | 无 pickle，读取不执行任何反序列化代码 |
| manifest.csv 列数 | 59 | 含执行期三列（内存/耗时/错误） |
| results_100.csv 列数 | 56 | manifest 去掉执行期三列 |
| 表行数 | 100 / 100 | 两份表各恰 100 行，键集合相等（V1 断言） |
| 三模态共同有效词数 valid_length | 逐样本一列 | text_word_valid ∧ word_audio_valid ∧ word_vision_valid |
