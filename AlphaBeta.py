def alphaBeta(depth, nodeIndex, isMaximizing, values, alpha, beta, maxDepth):
    if depth == maxDepth:  # If we reach a leaf node or the maximum depth
        return values[nodeIndex]

    if isMaximizing:
        best = float("-inf")  # Set the best value as the negative infinity

        # Exploring the left and right child nodes
        for i in range(2):
            val = alphaBeta(
                depth + +1, nodeIndex * 2 + i, False, values, alpha, beta, maxDepth
            )
            best = max(best, val)  # Choose the maximum value
            alpha = max(alpha, best)

            # Alpha Beta Pruning Condition
            if beta <= alpha:
                break

        return best

    else:
        best = float("inf")  # Set the best value equal to positive infinity

        # Explore left and right child nodes
        for i in range(2):
            val = alphaBeta(
                depth + 1, nodeIndex * 2 + i, True, values, alpha, beta, maxDepth
            )
            best = min(best, val)  # Choose the minimum value
            beta = min(beta, best)

            # Alpha Beta Pruning Condition
            if beta <= alpha:
                break

        return best


# Example Use Case
if __name__ == "__main__":
    values = [3, 4, 8, 2, -1, 8, 3, 6, 7, 4]  # Example leave nodes

    print(
        "The optimal value is:",
        alphaBeta(0, 0, True, values, float("-inf"), float("inf"), 3),
    )
