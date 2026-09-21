# 异常定义模块：统一描述请求、输入、推理和输出校验错误。
class CompetitionError(Exception):
    """Base class for expected pipeline failures."""


class InvalidRequestError(CompetitionError):
    pass


class InvalidInputError(CompetitionError):
    pass


class MissingSeriesError(CompetitionError):
    pass


class ModelInferenceError(CompetitionError):
    pass


class InvalidTaskResultError(CompetitionError):
    pass


class OutputValidationError(CompetitionError):
    pass
