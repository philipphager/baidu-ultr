from enum import IntEnum

TOKEN_OFFSET = 10

TENCENT_SPECIAL_TOKENS = {
    "PAD": 0,
    "SEP": 1,
    "CLS": 2,
    "MASK": 3,
}

BAIDU_SPECIAL_TOKENS = {
    "CLS": 0,
    "SEP": 1,
    "PAD": 2,
    "MASK": 3,
}

SEGMENT_TYPES = {
    "QUERY": 0,
    "TEXT": 1,
    "PAD": 1,  # See source code:
}


class QueryColumns(IntEnum):
    QID = 0
    QUERY = 1
    QUERY_REFORMULATION = 2


class TrainColumns(IntEnum):
    POS = 0
    URL_MD5 = 1
    TITLE = 2
    ABSTRACT = 3
    MULTIMEDIA_TYPE = 4
    CLICK = 5
    SKIP = 8
    SERP_HEIGHT = 9
    DISPLAYED_TIME = 10
    DISPLAYED_TIME_MIDDLE = 11
    FIRST_CLICK = 12
    DISPLAYED_COUNT = 13
    SERO_MAX_SHOW_HEIGHT = 14
    SLIPOFF_COUNT_AFTER_CLICK = 15
    DWELLING_TIME = 16
    DISPLAYED_TIME_TOP = 17
    SERO_TO_TOP = 18
    DISPLAYED_COUNT_TOP = 19
    DISPLAYED_COUNT_BOTTOM = 20
    SLIPOFF_COUNT = 21
    FINAL_CLICK = 23
    DISPLAYED_TIME_BOTTOM = 24
    CLICK_COUNT = 25
    DISPLAYED_COUNT_2 = 26
    LAST_CLICK = 28
    REVERSE_DISPLAY_COUNT = 29
    DISPLAYED_COUNT_MIDDLE = 30


class TestColumns(IntEnum):
    QUERY = 0
    TITLE = 1
    ABSTRACT = 2
    LABEL = 3
    BUCKET = 4
