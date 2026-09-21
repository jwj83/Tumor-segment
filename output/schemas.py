# 输出契约常量：定义预测、分割掩码和重复病例结果的字段要求。
CONTRACT_VERSION = "0.1"

PREDICTION_REQUIRED_KEYS = {
    "AccessionNumber",
    "IsNotHumanBodyProb",
    "IsStitchedProb",
    "ProcessingTime_ms",
    "SegmentationMaskURI",
    "Prediction",
    "Interpretation",
}

MASK_KEYS = {"core", "flair"}

DUPLICATE_KEYS = {"StudyUID", "StudyUID_dup", "PairProb"}
