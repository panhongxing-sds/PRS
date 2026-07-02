"""Math answer clustering via math_equal equivalence."""

from __future__ import annotations

from panda.grading.math_grader import extract_math_answer, math_equal


def _normalize_answer(text: str) -> str:
    return extract_math_answer(text).strip()


class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def cluster_answers(answers: list[str]) -> tuple[list[int], dict[int, int]]:
    """
    Cluster semantically equivalent math answers.

    Returns:
        labels: cluster id per answer (0..K-1)
        cluster_sizes: {cluster_id: count}
    """
    n = len(answers)
    if n == 0:
        return [], {}

    normed = [_normalize_answer(a) for a in answers]
    uf = _UnionFind(n)
    for i in range(n):
        for j in range(i + 1, n):
            if not normed[i] or not normed[j]:
                continue
            if math_equal(normed[i], normed[j]):
                uf.union(i, j)

    root_to_label: dict[int, int] = {}
    labels: list[int] = []
    for i in range(n):
        root = uf.find(i)
        if root not in root_to_label:
            root_to_label[root] = len(root_to_label)
        labels.append(root_to_label[root])

    sizes: dict[int, int] = {}
    for lab in labels:
        sizes[lab] = sizes.get(lab, 0) + 1
    return labels, sizes
