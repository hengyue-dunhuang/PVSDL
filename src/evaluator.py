"""
Evaluation Module
Calculates various classification metrics: Accuracy, Precision, Recall, F1-score, etc.
"""
import json
import numpy as np
from pathlib import Path
from src.utils import ensure_dir, format_timestamp, parse_model_response


class Evaluator:
    def __init__(self):
        """Initialize the evaluator"""
        pass
    
    def calculate_metrics(self, predictions, ground_truth):
        """
        Calculate classification metrics.
        
        Args:
            predictions: List of predicted labels (0 or 1)
            ground_truth: List of ground truth labels (0 or 1)
            
        Returns:
            dict: Dictionary containing various metrics
        """
        predictions = np.array(predictions)
        ground_truth = np.array(ground_truth)
        
        # Handle unparsable predictions marked as -1
        valid_mask = predictions != -1
        valid_predictions = predictions[valid_mask]
        valid_ground_truth = ground_truth[valid_mask]
        
        if len(valid_predictions) == 0:
            return {
                "error": "No valid prediction results",
                "total_samples": len(predictions),
                "valid_samples": 0,
                "invalid_samples": len(predictions)
            }
        
        # Calculate Confusion Matrix
        tp = np.sum((valid_predictions == 1) & (valid_ground_truth == 1))  # True Positive
        tn = np.sum((valid_predictions == 0) & (valid_ground_truth == 0))  # True Negative
        fp = np.sum((valid_predictions == 1) & (valid_ground_truth == 0))  # False Positive
        fn = np.sum((valid_predictions == 0) & (valid_ground_truth == 1))  # False Negative
        
        # Calculate individual metrics
        total = len(valid_predictions)
        accuracy = (tp + tn) / total if total > 0 else 0
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Accuracy per class
        clean_correct = tn
        clean_total = np.sum(valid_ground_truth == 0)
        clean_accuracy = clean_correct / clean_total if clean_total > 0 else 0
        
        dirty_correct = tp
        dirty_total = np.sum(valid_ground_truth == 1)
        dirty_accuracy = dirty_correct / dirty_total if dirty_total > 0 else 0
        
        metrics = {
            "total_samples": len(predictions),
            "valid_samples": len(valid_predictions),
            "invalid_samples": len(predictions) - len(valid_predictions),
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1_score, 4),
            "confusion_matrix": {
                "true_positive": int(tp),
                "true_negative": int(tn),
                "false_positive": int(fp),
                "false_negative": int(fn)
            },
            "per_class_accuracy": {
                "clean": round(clean_accuracy, 4),
                "dirty": round(dirty_accuracy, 4)
            },
            "balanced_accuracy": round((clean_accuracy + dirty_accuracy) / 2, 4)
        }
        
        return metrics
    
    def evaluate_predictions(self, raw_predictions):
        """
        Evaluate performance from raw prediction results.
        
        Args:
            raw_predictions: List of raw prediction results, each containing 'response' and 'ground_truth'
            
        Returns:
            dict: Evaluation metrics
        """
        predictions = []
        ground_truth = []
        parse_errors = []
        
        for idx, pred in enumerate(raw_predictions):
            if not pred.get('success', False):
                # API call failed
                predictions.append(-1)
                ground_truth.append(pred['image_info']['label'])
                parse_errors.append({
                    "index": idx,
                    "filename": pred['image_info']['filename'],
                    "error": pred.get('error', 'Unknown error')
                })
                continue
            
            # Parse model response
            parsed_label = parse_model_response(pred['response'])
            predictions.append(parsed_label)
            ground_truth.append(pred['image_info']['label'])
            
            if parsed_label == -1:
                parse_errors.append({
                    "index": idx,
                    "filename": pred['image_info']['filename'],
                    "response": pred['response'],
                    "error": "Failed to parse model response"
                })
        
        # Calculate metrics
        metrics = self.calculate_metrics(predictions, ground_truth)
        metrics['parse_errors'] = parse_errors
        metrics['parse_error_count'] = len(parse_errors)
        
        return metrics
    
    def save_metrics(self, metrics, output_path):
        """
        Save evaluation metrics to a JSON file.
        
        Args:
            metrics: Dictionary of evaluation metrics
            output_path: Path to the output file
        """
        ensure_dir(Path(output_path).parent)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Evaluation metrics saved to: {output_path}")
    
    def generate_report(self, metrics, detailed=True):
        """
        Generate a human-readable evaluation report.
        
        Args:
            metrics: Dictionary of evaluation metrics
            detailed: Whether to show detailed error information
            
        Returns:
            str: Formatted report text
        """
        report = []
        report.append("\n" + "=" * 60)
        report.append("EVALUATION REPORT")
        report.append("=" * 60)
        
        # Basic Statistics
        report.append(f"\nSample Statistics:")
        report.append(f"  Total Samples:   {metrics['total_samples']}")
        report.append(f"  Valid Samples:   {metrics['valid_samples']}")
        report.append(f"  Invalid Samples: {metrics['invalid_samples']}")
        
        if 'parse_error_count' in metrics:
            report.append(f"  Parse Errors:    {metrics['parse_error_count']}")
        
        # Main Metrics
        report.append(f"\nMain Metrics:")
        report.append(f"  Accuracy:          {metrics['accuracy']:.2%}")
        report.append(f"  Precision:         {metrics['precision']:.2%}")
        report.append(f"  Recall:            {metrics['recall']:.2%}")
        report.append(f"  F1-Score:          {metrics['f1_score']:.2%}")
        report.append(f"  Balanced Accuracy: {metrics['balanced_accuracy']:.2%}")
        
        # Per-class Accuracy
        report.append(f"\nPer-class Accuracy:")
        report.append(f"  Clean: {metrics['per_class_accuracy']['clean']:.2%}")
        report.append(f"  Dirty: {metrics['per_class_accuracy']['dirty']:.2%}")
        
        # Confusion Matrix
        cm = metrics['confusion_matrix']
        report.append(f"\nConfusion Matrix:")
        report.append(f"                 Predicted Clean   Predicted Dirty")
        report.append(f"  Actual Clean:  {cm['true_negative']:>15}   {cm['false_positive']:>15}")
        report.append(f"  Actual Dirty:  {cm['false_negative']:>15}   {cm['true_positive']:>15}")
        
        # Detailed Error Information
        if detailed and 'parse_errors' in metrics and len(metrics['parse_errors']) > 0:
            report.append(f"\nParse Error Details:")
            for error in metrics['parse_errors'][:10]:  # Show top 10
                report.append(f"  [{error['index']}] {error['filename']}")
                report.append(f"      {error['error']}")
                if 'response' in error:
                    report.append(f"      Response: {error['response'][:50]}...")
        
        report.append("=" * 60 + "\n")
        
        return "\n".join(report)


if __name__ == "__main__":
    # Test code
    evaluator = Evaluator()
    
    # Simulated prediction results
    test_predictions = [0, 1, 0, 1, 1, 0, 1, 0, 0, 1]
    test_ground_truth = [0, 1, 0, 1, 0, 0, 1, 1, 0, 1]
    
    print("Testing Evaluator...")
    metrics = evaluator.calculate_metrics(test_predictions, test_ground_truth)
    
    print("\nCalculated Metrics:")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    
    print(evaluator.generate_report(metrics))