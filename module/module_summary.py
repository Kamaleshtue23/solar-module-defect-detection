from collections import Counter


def build_summary(results):

    total = len(results)

    defective = sum(
        r["defective"]
        for r in results
    )

    healthy = total - defective

    defect_counter = Counter()

    for r in results:

        for d in r["defects"]:

            defect_counter[d["name"]] += 1

    return {
        "total": total,
        "defective": defective,
        "healthy": healthy,
        "defect_counts": dict(defect_counter)
    }