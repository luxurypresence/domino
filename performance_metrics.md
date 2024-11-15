
# Evaluating the Performance of a Property Recommendation System

To evaluate the effectiveness of the property recommendation system, we can use several relevance metrics, including **Precision@K**, **Recall@K**, and **Mean Average Precision (MAP)**. This document explains each metric and provides Python code examples for calculating them.

---

## 1. Precision@K

### Definition
Measures the proportion of relevant items in the top-K results.

**Formula**:
Precision@K = Number of Relevant Items in Top-K Predictions}/K


### Example
- **Predicted Properties**: `["2", "3", "4", "5", "6"]`
- **Ground-Truth Relevant Properties**: `{"3", "5", "7"}`

**Calculation**:
- Relevant items in top-5 predictions: `["3", "5"]` (2 items are relevant).
- Precision@5:
  Precision@5 = 2/5 = 0.4
  

### Python Code
```python
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
    return relevant_at_k / k

# Example usage
relevant_properties = {"3", "5", "7"}
predicted_properties = ["2", "3", "4", "5", "6"]
k = 5
print(f"Precision@{k}: {precision_at_k(relevant_properties, predicted_properties, k)}")
```

---

## 2. Recall@K

### Definition
Measures how many of the relevant items are returned in the top-K predictions.

**Formula**:
Recall@K=(Number of Relevant Items in Top-K Predictions)/(Total Number of Relevant Items)
### Example
Using the same data:
- Relevant items in top-5 predictions: `["3", "5"]` (2 items are relevant).
- Total relevant items: `3` (from `{"3", "5", "7"}`).
- Recall@5:
  Recall@5 = 2/3 = 0.67

### Python Code
```python
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
    return relevant_at_k / len(relevant)

# Example usage
print(f"Recall@{k}: {recall_at_k(relevant_properties, predicted_properties, k)}")
```

---

## 3. Mean Average Precision (MAP)

### Definition
Aggregates precision across all relevant items in the predictions, providing a single score for the entire ranking.

**Formula**:
\[
\text{MAP} = \frac{\sum_{i=1}^{N} \text{Precision@i (if relevant)}}{\text{Number of Relevant Items}}
\]

### Example
Using the same data:
- Predictions: `["2", "3", "4", "5", "6"]`.
- Ground-truth relevant items: `{"3", "5", "7"}`.

**Step-by-Step Calculation**:
1. For position 2 (property `3` is relevant): Precision@2 = \( \frac{1}{2} \).
2. For position 4 (property `5` is relevant): Precision@4 = \( \frac{2}{4} = 0.5 \).

**MAP**:
\[
\text{MAP} = \frac{1/2 + 2/4}{3} = \frac{0.5 + 0.5}{3} \approx 0.333
\]

### Python Code
```python
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

    if relevant_count == 0:
        return 0.0
    return average_precision / len(relevant)

# Example usage
print(f"MAP: {mean_average_precision(relevant_properties, predicted_properties)}")
```

---

## 4. Applying Metrics to Your Model

### Example for Property ID `1`
1. **Retrieve Predictions**:
   ```python
   predicted_properties = model.get_similar_properties("1")
   ```
2. **Ground-Truth Relevant Properties**:
   ```python
   relevant_properties = {"3", "5", "7"}  # Replace with actual ground-truth data.
   ```
3. **Calculate Metrics**:
   ```python
   k = 5
   print(f"Precision@{k}: {precision_at_k(relevant_properties, predicted_properties, k)}")
   print(f"Recall@{k}: {recall_at_k(relevant_properties, predicted_properties, k)}")
   print(f"MAP: {mean_average_precision(relevant_properties, predicted_properties)}")
   ```

---

## 5. Aggregating Metrics Across Multiple Properties

For a complete evaluation, run the metrics for multiple property IDs and average the results.

### Example
```python
property_ids = ["1", "2", "3"]  # List of property IDs to evaluate
overall_precision = []
overall_recall = []
overall_map = []

for pid in property_ids:
    predicted = model.get_similar_properties(pid)
    relevant = ground_truth[pid]  # Dictionary of ground-truth relevant properties
    overall_precision.append(precision_at_k(relevant, predicted, k=5))
    overall_recall.append(recall_at_k(relevant, predicted, k=5))
    overall_map.append(mean_average_precision(relevant, predicted))

# Calculate averages
print(f"Avg Precision@5: {sum(overall_precision) / len(overall_precision)}")
print(f"Avg Recall@5: {sum(overall_recall) / len(overall_recall)}")
print(f"Avg MAP: {sum(overall_map) / len(overall_map)}")
```

---

## 6. Conclusion

- Use **Precision@K** and **Recall@K** to measure the relevance of the top-K results.
- Use **MAP** to evaluate the ranking quality across all relevant properties.
- Apply these metrics to individual properties or aggregate across multiple properties to assess overall performance.
```