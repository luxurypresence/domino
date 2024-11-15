class RelevanceMetricsCalculator:
    """
    A class to calculate relevance metrics for evaluating recommendation systems.
    Includes Precision@K, Recall@K, and Mean Average Precision (MAP).
    """

    @staticmethod
    def precision_at_k(relevant, predicted, k):
        """
        Compute Precision@K
        :param relevant: Set of ground-truth relevant items
        :param predicted: List of predicted items
        :param k: Number of top results to evaluate
        :return: Precision@K
        """
        predicted_at_k = predicted[:k]
        relevant_at_k = len(set(predicted_at_k) & relevant)
        return relevant_at_k / k if k > 0 else 0.0

    @staticmethod
    def recall_at_k(relevant, predicted, k):
        """
        Compute Recall@K
        :param relevant: Set of ground-truth relevant items
        :param predicted: List of predicted items
        :param k: Number of top results to evaluate
        :return: Recall@K
        """
        predicted_at_k = predicted[:k]
        relevant_at_k = len(set(predicted_at_k) & relevant)
        return relevant_at_k / len(relevant) if relevant else 0.0

    @staticmethod
    def mean_average_precision(relevant, predicted):
        """
        Compute Mean Average Precision (MAP)
        :param relevant: Set of ground-truth relevant items
        :param predicted: List of predicted items
        :return: MAP score
        """
        average_precision = 0.0
        relevant_count = 0

        for i, p in enumerate(predicted, start=1):
            if p in relevant:
                relevant_count += 1
                precision = relevant_count / i  # Precision@i
                average_precision += precision

        return average_precision / len(relevant) if relevant else 0.0

    @staticmethod
    def evaluate_metrics(relevant, predicted, k):
        """
        Evaluate Precision@K, Recall@K, and MAP for a single property.
        :param relevant: Set of ground-truth relevant items
        :param predicted: List of predicted items
        :param k: Number of top results to evaluate
        :return: Dictionary with Precision@K, Recall@K, and MAP
        """
        precision = RelevanceMetricsCalculator.precision_at_k(relevant, predicted, k)
        recall = RelevanceMetricsCalculator.recall_at_k(relevant, predicted, k)
        map_score = RelevanceMetricsCalculator.mean_average_precision(relevant, predicted)

        return {
            f"Precision@{k}": precision,
            f"Recall@{k}": recall,
            "MAP": map_score
        }

    @staticmethod
    def evaluate_multiple(properties, ground_truth, model, k):
        """
        Evaluate relevance metrics across multiple properties.
        :param properties: List of property IDs to evaluate
        :param ground_truth: Dictionary with ground-truth relevant items for each property
        :param model: Model object that provides `get_similar_properties(property_id)` method
        :param k: Number of top results to evaluate
        :return: Averages of Precision@K, Recall@K, and MAP across all properties
        """
        overall_precision = []
        overall_recall = []
        overall_map = []

        for pid in properties:
            predicted = model.get_similar_properties(pid)
            relevant = ground_truth.get(pid, set())  # Default to an empty set if no ground-truth exists

            overall_precision.append(RelevanceMetricsCalculator.precision_at_k(relevant, predicted, k))
            overall_recall.append(RelevanceMetricsCalculator.recall_at_k(relevant, predicted, k))
            overall_map.append(RelevanceMetricsCalculator.mean_average_precision(relevant, predicted))

        return {
            f"Avg Precision@{k}": sum(overall_precision) / len(properties),
            f"Avg Recall@{k}": sum(overall_recall) / len(properties),
            "Avg MAP": sum(overall_map) / len(properties)
        }


# evaluate for single property
relevant_properties = {"3", "5", "7"}  # Ground-truth relevant items for property ID 1
predicted_properties = ["2", "3", "4", "5", "6"]  # Predicted results
k = 5

# Calculate metrics
calculator = RelevanceMetricsCalculator()
metrics = calculator.evaluate_metrics(relevant_properties, predicted_properties, k)
print(metrics)

# Example data
# property_ids = ["1", "2", "3"]  # List of property IDs
# ground_truth = {
#     "1": {"3", "5", "7"},
#     "2": {"10", "12", "15"},
#     "3": {"20", "25", "30"}
# }  # Ground-truth relevant items for each property
# model=
# aggregated_metrics = RelevanceMetricsCalculator().evaluate_multiple(property_ids, ground_truth, model, k)
# print(aggregated_metrics)