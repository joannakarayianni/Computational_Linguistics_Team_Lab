from collections import defaultdict

class EvaluationMetrics:
    def __init__(self, true_labels, predicted_labels, labels):
        self.true_labels = true_labels
        self.predicted_labels = predicted_labels
        self.labels = labels

    def compute_metrics(self):
        # Initialize dictionaries to store true positives, false positives, and false negatives for each label
        tp = defaultdict(int)
        fp = defaultdict(int)
        fn = defaultdict(int)

        # Compute true positives, false positives, and false negatives for each label
        for true_label, predicted_label in zip(self.true_labels, self.predicted_labels):
            for label in self.labels:
                if label in true_label and label in predicted_label:
                    tp[label] += 1
                elif label not in true_label and label in predicted_label:
                    fp[label] += 1
                elif label in true_label and label not in predicted_label:
                    fn[label] += 1

        # Compute precision, recall, and F-score for each label
        precision = {}
        recall = {}
        f_score = {}
        for label in self.labels:
            precision[label] = tp[label] / (tp[label] + fp[label]) if (tp[label] + fp[label]) > 0 else 0
            recall[label] = tp[label] / (tp[label] + fn[label]) if (tp[label] + fn[label]) > 0 else 0
            f_score[label] = (2 * precision[label] * recall[label]) / (precision[label] + recall[label]) if (precision[label] + recall[label]) > 0 else 0

        return precision, recall, f_score
    
    # Testing the evaluation metrics
def test_evaluation(true_labels, predicted_labels, labels):
    eval_metrics = EvaluationMetrics(true_labels, predicted_labels, labels)
    precision, recall, f_score = eval_metrics.compute_metrics()

    print("Precision:")
    print(precision)
    print("Recall:")
    print(recall)
    print("F-score:")
    print(f_score)

if __name__ == "__main__":
    test_evaluation()
