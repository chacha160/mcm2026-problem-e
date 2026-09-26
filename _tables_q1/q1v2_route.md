| 判定类别 | 机器可判 | 本批条数 | 判据 |
|---|---|---|---|
| TRI_MODAL_WORD_VALID | 否——含人工听辨结论 | 0 | 要求「已确认文本内容与讲话对应」，属人工听辨结论；机器只能判区间合法 |
| TRI_MODAL_WITH_AUDIO_CONTENT_INVALID | 是 | 2 | 时间轴可信且 int16 波形逐采样全零 |
| AV_VALID_TEXT_UNALIGNED | 否——含人工听辨结论 | 0 | 要求「已确认文字与讲话明显不对应」，属人工听辨结论；ASR 距离与对齐概率反推不出 |
| UNCERTAIN_REVIEW | 是 | 98 | 其余情况（兜底） |
| EXTRACTION_OR_TIMELINE_ANOMALY | 是 | 0 | 身份/解码/共享时间轴不可靠 |
