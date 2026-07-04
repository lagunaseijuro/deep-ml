import heapq


def zero_bucket_assignment(sizes: list, num_ranks: int) -> list:
    """
    Assign whole parameter matrices to ranks to balance loads.

    Args:
        sizes: list of parameter matrix sizes (number of elements).
        num_ranks: number of ZeRO ranks.

    Returns:
        List of length num_ranks with the total size assigned to each rank.
    """
    sizes.sort(reverse=True)
    heap = [0] * num_ranks
    heapq.heapify(heap)

    for idx in range(len(sizes)):
        el = heapq.heappop(heap)
        el += sizes[idx]
        heapq.heappush(heap, el)

    return sorted(list(heap), reverse=True)




