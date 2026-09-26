| 字段 | 取值分布（本批 100 条） | 语义 |
|---|---|---|
| alignment_granularity | word 97、clip 3 | 该样本的时序组织粒度（词级 / 片段级） |
| text_av_time_mapping_status | word_partial 50、word_valid 47、clip_only 3 | 结构闸门（机器可判）：官方词的时间区间够不够用 |
| text_audio_correspondence | not_asserted 98、no_speech 2 | 内容确认：文本与实际讲话是否对应；本套代码一律 not_asserted，唯一例外是位精确可证的数字静音 |
| word_time_basis | own_forced_alignment 97、none 3 | 词区间来源；own_forced_alignment = 至少一个词的区间来自本次冻结对齐运行 |
| content_assertion | 取 1 的条数 0 | uint8；1 仅当存在外部人工证据，本套恒为 0 |
| audio_speech_valid | -1 98、0 2 | 有语音的证据等级：0 = 已确认数字静音，−1 = 证据不足。**从不取 1** |
