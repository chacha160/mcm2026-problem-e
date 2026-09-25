| 异常代码 | 模态 | 条数 | 处置 |
|---|---|---|---|
| A6_audio_shorter_than_video | audio | 50 | 保留并标注 |
| A7_audio_valid_slots_lt_50 | audio | 46 | 保留并标注 |
| VFR_frame_gap | vision | 28 | 保留并标注 |
| LONG_PAUSE | audio | 6 | 保留并标注 |
| A8_pts_beyond_axis | vision | 5 | 保留并标注 |
| A4_vision_face_partial | vision | 4 | 保留并标注 |
| A1_audio_digital_silence | audio | 2 | 保留并标注 |
| A3_vision_face_all_fail | vision | 2 | 保留并标注 |
| A5_vision_energy_zero_but_valid | vision | 2 | 保留并标注 |
| A2_vad_zero_but_energy | audio | 1 | 保留并标注 |
| D1_DURATION_FLOAT32 | all | 1 | 已修复（本轮） |
| D3_FRAMES_TABLE_KEYED_READER | vision | 1 | 已修复（本轮） |
| D2_OFFICIAL_ID_FORMAT | all | 1 | 已修复（本轮） |
| R1_FILL_MISSING_FACES_INPLACE | vision | 1 | 已识别风险，本批结果不受影响 |
| R2_POOLER_RANDOM_INIT | text | 1 | 已解释，不影响特征 |
