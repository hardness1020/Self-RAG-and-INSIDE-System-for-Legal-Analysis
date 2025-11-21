"""
INSIDE Evaluation Module

Evaluation metrics specific to INSIDE functionality:
1. EigenScore calibration and threshold analysis
2. Intent detection accuracy
3. Hallucination detection performance (precision, recall, F1)
4. Intent-aware retrieval quality
5. Feature clipping effectiveness
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Optional, Tuple
from sklearn.metrics import precision_recall_curve, roc_curve, auc, classification_report
import pandas as pd


class EigenScoreEvaluator:
    """
    Evaluates EigenScore-based hallucination detection.

    Metrics:
    - Precision, Recall, F1 at various thresholds
    - ROC curve and AUC
    - Calibration analysis
    - Per-intent performance
    """

    def __init__(self):
        self.results = []

    def add_result(
        self,
        eigenscore: float,
        is_hallucination: bool,
        query_intent: Optional[str] = None,
        metadata: Optional[Dict] = None
    ):
        """
        Add a detection result for evaluation.

        Args:
            eigenscore: Computed EigenScore
            is_hallucination: Ground truth label
            query_intent: Query intent (optional)
            metadata: Additional metadata
        """
        self.results.append({
            'eigenscore': eigenscore,
            'is_hallucination': is_hallucination,
            'intent': query_intent,
            'metadata': metadata or {}
        })

    def compute_metrics(
        self,
        threshold: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Compute detection metrics.

        Args:
            threshold: EigenScore threshold (higher = hallucination)
                      If None, finds optimal threshold

        Returns:
            Dictionary with metrics
        """
        if not self.results:
            return {}

        eigenscores = np.array([r['eigenscore'] for r in self.results])
        labels = np.array([r['is_hallucination'] for r in self.results])

        # Find optimal threshold if not provided
        if threshold is None:
            # Threshold: hallucinations have higher EigenScore
            precision, recall, thresholds = precision_recall_curve(labels, eigenscores)
            f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
            optimal_idx = np.argmax(f1_scores)
            threshold = thresholds[optimal_idx]

        # Compute predictions
        predictions = eigenscores > threshold

        # Metrics
        tp = np.sum((predictions == 1) & (labels == 1))
        fp = np.sum((predictions == 1) & (labels == 0))
        tn = np.sum((predictions == 0) & (labels == 0))
        fn = np.sum((predictions == 0) & (labels == 1))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / len(labels)

        return {
            'threshold': threshold,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': accuracy,
            'tp': int(tp),
            'fp': int(fp),
            'tn': int(tn),
            'fn': int(fn)
        }

    def plot_roc_curve(self, save_path: Optional[str] = None):
        """Plot ROC curve for EigenScore detection."""
        if not self.results:
            print("No results to plot")
            return

        eigenscores = np.array([r['eigenscore'] for r in self.results])
        labels = np.array([r['is_hallucination'] for r in self.results])

        # ROC curve (higher scores indicate hallucination)
        fpr, tpr, _ = roc_curve(labels, eigenscores)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'EigenScore (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve: EigenScore Hallucination Detection')
        plt.legend()
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        return roc_auc

    def plot_eigenscore_distribution(
        self,
        save_path: Optional[str] = None
    ):
        """Plot EigenScore distribution for hallucinated vs factual."""
        if not self.results:
            print("No results to plot")
            return

        eigenscores = np.array([r['eigenscore'] for r in self.results])
        labels = np.array([r['is_hallucination'] for r in self.results])

        hallucinated_scores = eigenscores[labels == 1]
        factual_scores = eigenscores[labels == 0]

        plt.figure(figsize=(10, 6))
        plt.hist(factual_scores, bins=30, alpha=0.5, label='Factual', color='green')
        plt.hist(hallucinated_scores, bins=30, alpha=0.5, label='Hallucinated', color='red')
        plt.xlabel('EigenScore')
        plt.ylabel('Frequency')
        plt.title('EigenScore Distribution: Factual vs Hallucinated')
        plt.legend()
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def per_intent_analysis(self) -> pd.DataFrame:
        """Analyze performance per query intent."""
        if not self.results:
            return pd.DataFrame()

        # Group by intent
        intent_metrics = []

        intents = set(r['intent'] for r in self.results if r['intent'])

        for intent in intents:
            intent_results = [r for r in self.results if r['intent'] == intent]

            eigenscores = np.array([r['eigenscore'] for r in intent_results])
            labels = np.array([r['is_hallucination'] for r in intent_results])

            # Find optimal threshold for this intent
            precision, recall, thresholds = precision_recall_curve(labels, eigenscores)
            f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
            optimal_idx = np.argmax(f1_scores)
            threshold = thresholds[optimal_idx] if len(thresholds) > 0 else 0

            predictions = eigenscores > threshold

            tp = np.sum((predictions == 1) & (labels == 1))
            fp = np.sum((predictions == 1) & (labels == 0))
            tn = np.sum((predictions == 0) & (labels == 0))
            fn = np.sum((predictions == 0) & (labels == 1))

            precision_val = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall_val = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision_val * recall_val / (precision_val + recall_val) if (precision_val + recall_val) > 0 else 0

            intent_metrics.append({
                'intent': intent,
                'num_samples': len(intent_results),
                'optimal_threshold': threshold,
                'precision': precision_val,
                'recall': recall_val,
                'f1': f1,
                'mean_eigenscore': np.mean(eigenscores),
                'std_eigenscore': np.std(eigenscores)
            })

        return pd.DataFrame(intent_metrics)


class IntentDetectionEvaluator:
    """
    Evaluates intent detection accuracy.
    """

    def __init__(self):
        self.predictions = []
        self.ground_truth = []

    def add_result(self, predicted_intent: str, true_intent: str):
        """Add a prediction result."""
        self.predictions.append(predicted_intent)
        self.ground_truth.append(true_intent)

    def compute_metrics(self) -> Dict[str, any]:
        """Compute intent detection metrics."""
        if not self.predictions:
            return {}

        # Overall accuracy
        correct = sum(p == t for p, t in zip(self.predictions, self.ground_truth))
        accuracy = correct / len(self.predictions)

        # Per-class metrics
        report = classification_report(
            self.ground_truth,
            self.predictions,
            output_dict=True
        )

        return {
            'accuracy': accuracy,
            'classification_report': report,
            'num_samples': len(self.predictions)
        }

    def confusion_matrix_plot(self, save_path: Optional[str] = None):
        """Plot confusion matrix."""
        from sklearn.metrics import confusion_matrix
        import seaborn as sns

        cm = confusion_matrix(self.ground_truth, self.predictions)
        labels = sorted(set(self.ground_truth + self.predictions))

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=labels, yticklabels=labels)
        plt.xlabel('Predicted Intent')
        plt.ylabel('True Intent')
        plt.title('Intent Detection Confusion Matrix')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()


class RetrievalQualityEvaluator:
    """
    Evaluates intent-aware retrieval quality.

    Metrics:
    - Per-intent retrieval accuracy
    - Diversity scores
    - Contrast effectiveness
    """

    def __init__(self):
        self.results = []

    def add_result(
        self,
        query_intent: str,
        retrieved_docs: List[str],
        relevant_docs: List[str],
        diversity_score: Optional[float] = None
    ):
        """Add retrieval result."""
        self.results.append({
            'intent': query_intent,
            'retrieved': set(retrieved_docs),
            'relevant': set(relevant_docs),
            'diversity': diversity_score
        })

    def compute_metrics(self) -> Dict[str, float]:
        """Compute retrieval metrics."""
        if not self.results:
            return {}

        # Overall precision and recall
        total_retrieved = sum(len(r['retrieved']) for r in self.results)
        total_relevant_retrieved = sum(len(r['retrieved'] & r['relevant']) for r in self.results)
        total_relevant = sum(len(r['relevant']) for r in self.results)

        precision = total_relevant_retrieved / total_retrieved if total_retrieved > 0 else 0
        recall = total_relevant_retrieved / total_relevant if total_relevant > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        # Average diversity
        diversities = [r['diversity'] for r in self.results if r['diversity'] is not None]
        avg_diversity = np.mean(diversities) if diversities else None

        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'avg_diversity': avg_diversity,
            'num_queries': len(self.results)
        }

    def per_intent_metrics(self) -> pd.DataFrame:
        """Compute metrics per intent."""
        if not self.results:
            return pd.DataFrame()

        intent_metrics = []
        intents = set(r['intent'] for r in self.results)

        for intent in intents:
            intent_results = [r for r in self.results if r['intent'] == intent]

            retrieved = sum(len(r['retrieved']) for r in intent_results)
            relevant_retrieved = sum(len(r['retrieved'] & r['relevant']) for r in intent_results)
            relevant = sum(len(r['relevant']) for r in intent_results)

            precision = relevant_retrieved / retrieved if retrieved > 0 else 0
            recall = relevant_retrieved / relevant if relevant > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            intent_metrics.append({
                'intent': intent,
                'num_queries': len(intent_results),
                'precision': precision,
                'recall': recall,
                'f1': f1
            })

        return pd.DataFrame(intent_metrics)


def run_comprehensive_evaluation(
    eigenscore_results: List[Dict],
    intent_results: List[Tuple[str, str]],
    retrieval_results: List[Dict],
    output_dir: str = 'evaluation_results'
) -> Dict[str, any]:
    """
    Run comprehensive INSIDE evaluation.

    Args:
        eigenscore_results: List of dicts with 'eigenscore', 'is_hallucination', 'intent'
        intent_results: List of (predicted, true) intent pairs
        retrieval_results: List of dicts with retrieval quality data
        output_dir: Directory to save plots

    Returns:
        Dictionary with all evaluation metrics
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    # EigenScore evaluation
    eigenscore_eval = EigenScoreEvaluator()
    for result in eigenscore_results:
        eigenscore_eval.add_result(
            eigenscore=result['eigenscore'],
            is_hallucination=result['is_hallucination'],
            query_intent=result.get('intent')
        )

    eigenscore_metrics = eigenscore_eval.compute_metrics()
    eigenscore_eval.plot_roc_curve(f'{output_dir}/eigenscore_roc.png')
    eigenscore_eval.plot_eigenscore_distribution(f'{output_dir}/eigenscore_dist.png')
    per_intent_df = eigenscore_eval.per_intent_analysis()

    # Intent detection evaluation
    intent_eval = IntentDetectionEvaluator()
    for pred, true in intent_results:
        intent_eval.add_result(pred, true)

    intent_metrics = intent_eval.compute_metrics()
    intent_eval.confusion_matrix_plot(f'{output_dir}/intent_confusion.png')

    # Retrieval quality evaluation
    retrieval_eval = RetrievalQualityEvaluator()
    for result in retrieval_results:
        retrieval_eval.add_result(
            query_intent=result['intent'],
            retrieved_docs=result['retrieved'],
            relevant_docs=result['relevant'],
            diversity_score=result.get('diversity')
        )

    retrieval_metrics = retrieval_eval.compute_metrics()
    retrieval_per_intent = retrieval_eval.per_intent_metrics()

    # Compile results
    results = {
        'eigenscore': eigenscore_metrics,
        'eigenscore_per_intent': per_intent_df.to_dict() if not per_intent_df.empty else {},
        'intent_detection': intent_metrics,
        'retrieval': retrieval_metrics,
        'retrieval_per_intent': retrieval_per_intent.to_dict() if not retrieval_per_intent.empty else {}
    }

    # Save summary
    with open(f'{output_dir}/evaluation_summary.txt', 'w') as f:
        f.write("INSIDE Evaluation Summary\n")
        f.write("=" * 80 + "\n\n")

        f.write("EigenScore Hallucination Detection:\n")
        for key, value in eigenscore_metrics.items():
            f.write(f"  {key}: {value}\n")

        f.write("\nIntent Detection:\n")
        f.write(f"  Accuracy: {intent_metrics.get('accuracy', 0):.4f}\n")

        f.write("\nRetrieval Quality:\n")
        for key, value in retrieval_metrics.items():
            f.write(f"  {key}: {value}\n")

    print(f"Evaluation results saved to {output_dir}/")

    return results
