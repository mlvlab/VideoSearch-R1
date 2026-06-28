import json
from dataclasses import dataclass
from typing import List


@dataclass
class CorpusItem:
    media_id: str
    media_type: str  # image | video
    path: str


@dataclass
class TrainItem:
    qid: str
    query: str
    pos_id: str


@dataclass
class SegmentItem:
    doc_id: str
    video_id: str
    video_path: str
    start: float
    end: float
    duration: float


@dataclass
class QueryItem:
    qid: str
    query: str
    pos_doc_id: str
    weight: float = 1.0


def load_corpus(path: str) -> List[CorpusItem]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            j = json.loads(line)
            out.append(CorpusItem(**j))
    return out


def load_train(path: str) -> List[TrainItem]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            j = json.loads(line)
            out.append(TrainItem(**j))
    return out


def load_segments(path: str) -> List[SegmentItem]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            j = json.loads(line)
            out.append(SegmentItem(**j))
    return out


def load_queries(path: str) -> List[QueryItem]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            j = json.loads(line)
            out.append(QueryItem(**j))
    return out
