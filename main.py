import argparse
import json
import csv
import os
from datetime import datetime
from pathlib import Path

from src.data_processor import DatasetProcessor
from src.vlm_client import VLMClient
from src.utils import ensure_dir, format_timestamp
from config.prompts import get_prompt, list_prompts, PROMPTS

def load_model_config():
    """Load model configuration"""
    with open('config/models.json', 'r', encoding='utf-8') as f:
        return json.load(f)


def list_available_models():
    """List all available models"""
    config = load_model_config()
    print("\nAvailable VLM Models:")
    print("=" * 80)
    for model in config['models']:
        print(f"\nID: {model['id']}")
        print(f"Name: {model['name']}")
        print(f"Provider: {model['provider']}")
        print(f"Description: {model['description']}")
    print("\nDefault Models: " + ", ".join(config['default_models']))
    print("=" * 80)


def build_dataset_index(args):
    """Build dataset index"""
    print("\n=== Building Dataset Index ===")
    processor = DatasetProcessor(
        images_dir=args.images_dir,
        index_file=args.index_file
    )
    processor.build_index(force_rebuild=args.force_rebuild)


def calculate_metrics(predictions, sampled_images):
    """
    Calculate binary classification metrics (TP/TN/FP/FN)
    Solar panel dust detection:
    - Positive: dirty, Negative: clean
    Fixes:
    1. Handle ground truth labels (integer 0/1 or string)
    2. Extract labels using the 'response' field from prediction results
    """
    tp = tn = fp = fn = 0
    failed_count = 0
    
    print(f"\n   📝 Metric Calculation Debug Log ({len(predictions)} total predictions):")
    for idx, (pred, img) in enumerate(zip(predictions, sampled_images)):
        # Skip and log failed predictions
        if not pred['success']:
            failed_count += 1
            print(f"    Prediction {idx+1}: Failed (success=False)")
            continue
        
        # -------------------------- Fix 1: Handle Ground Truth (Integer 0/1) --------------------------
        true_label = img.get('label', img.get('ground_truth', 'unknown'))
        
        # Process ground truth: 0=clean, 1=dirty
        if isinstance(true_label, int):
            true_is_dirty = true_label
        elif isinstance(true_label, str):
            true_is_dirty = 1 if 'dirty' in true_label.lower() else 0
        else:
            true_is_dirty = 1 if 'dirty' in str(true_label).lower() else 0
        
        # -------------------------- Fix 2: Extract from 'response' field --------------------------
        # The prediction results contain a 'response' field (e.g., 'response': 'clean')
        # Defensive check for None values: use 'unknown' if response is None
        raw_res = pred.get('response')
        pred_label = str(raw_res if raw_res is not None else 'unknown').lower()
        
        # Process predicted label: 'dirty' -> 1, otherwise -> 0
        pred_is_dirty = 1 if 'dirty' in pred_label else 0
        
        # Print debug info (showing raw and converted values)
        print(f"    Pred {idx+1}: True={true_label} | Pred={pred_label} | "
              f"True_Is_Dirty={true_is_dirty} | Pred_Is_Dirty={pred_is_dirty}")
        
        # Calculate metrics
        if true_is_dirty == 1 and pred_is_dirty == 1:
            tp += 1
        elif true_is_dirty == 0 and pred_is_dirty == 0:
            tn += 1
        elif true_is_dirty == 0 and pred_is_dirty == 1:
            fp += 1
        elif true_is_dirty == 1 and pred_is_dirty == 0:
            fn += 1
    
    # Print statistics
    print(f"  📊 Stats: Failed={failed_count} | TP={tp} | TN={tn} | FP={fp} | FN={fn}")
    
    return tp, tn, fp, fn


def write_to_csv(csv_path, data_row):
    """
    Fixed: CSV uses comma delimiter
    """
    headers = [
        'true_positive', 'true_negative', 'false_positive', 'false_negative',
        'model_id', 'prompt_id', 'run_num', 'samples', 'timestamp'
    ]
    
    file_exists = os.path.exists(csv_path)
    
    try:
        with open(csv_path, 'a', newline='', encoding='utf-8') as f:
            # Fixed: Changed delimiter='\t' to delimiter=','
            writer = csv.DictWriter(f, fieldnames=headers, delimiter=',')
            
            if not file_exists:
                writer.writeheader()
            
            writer.writerow(data_row)
        
        print(f"✅ Data appended to CSV: {csv_path} (Run: {data_row['run_num']})")
    
    except Exception as e:
        print(f"❌ Failed to write CSV: {str(e)}")


def run_repeated_test(args):
    """
    Runs test N times for a single model + single prompt, writing to CSV in real-time.
    """
    print("\n" + "=" * 80)
    print(f"VLM Solar Panel Dust Detection - Repeated Test Mode")
    print(f"Model: {args.single_model} | Prompt: {args.single_prompt}")
    print(f"Repeats: {args.repeat} | Samples per run: {args.samples}")
    print("=" * 80)
    
    # 1. Load Dataset
    print("\nStep 1/3: Loading dataset...")
    processor = DatasetProcessor(
        images_dir=args.images_dir,
        index_file=args.index_file
    )
    
    try:
        processor.load_index()
    except FileNotFoundError:
        print("  Dataset index not found, building...")
        processor.build_index()
    
    # Show dataset stats
    stats = processor.get_statistics()
    print(f"  Dataset Stats: Total {stats['total']} (Clean: {stats['clean']}, Dirty: {stats['dirty']})")
    
    # 2. Load Config
    print("\nStep 2/3: Loading test configuration...")
    model_config = load_model_config()
    model_map = {m['id']: m['name'] for m in model_config['models']}
    
    # Validate model and prompt
    if args.single_model not in model_map:
        raise ValueError(f"Model {args.single_model} does not exist! Available: {list(model_map.keys())}")
    
    try:
        prompt_text = get_prompt(args.single_prompt)
    except ValueError as e:
        raise ValueError(f"Prompt Error: {e}")
    
    model_name = model_map[args.single_model]
    print(f"  Model Name: {model_name}")
    print(f"  Prompt Content: {prompt_text[:50]}...")
    
    # 3. Initialize VLM Client
    print("\nStep 3/3: Starting repeated tests...")
    client = VLMClient()
    ensure_dir('results/raw_predictions')
    
    # CSV path
    csv_path = "vlm_experiment_results.csv"

    # Pre-fetch image list if testing all images
    all_images = None
    if args.all_images:
        all_images = processor.dataset_index['images']
        print(f"  Mode: Testing all images (Total: {len(all_images)})")
    
    # Loop for repeated tests
    for run_num in range(1, args.repeat + 1):
        print("\n" + "-" * 80)
        print(f"Repeat Run {run_num}/{args.repeat}")
        print("-" * 80)
        
        try:
            if args.all_images:
                sampled_images = all_images
                current_seed = args.seed
                print(f"  Using all images: {len(sampled_images)}")
            else:
                # Use different seed per run to avoid identical sampling
                current_seed = args.seed + run_num - 1
                
                # Resample dataset
                sampled_images = processor.sample_dataset(
                    n_samples=args.samples,
                    balanced=True,
                    random_seed=current_seed
                )
                print(f"  Sampling complete: {len(sampled_images)} images (Seed: {current_seed})")
            
            # Print ground truth for debugging
            print(f"  📌 Ground Truth Labels: {[img.get('label', img.get('ground_truth', 'unknown')) for img in sampled_images]}")
            
            # Execute VLM Query
            predictions = client.batch_query(
                image_list=sampled_images,
                prompt=prompt_text,
                model_name=model_name,
                verbose=args.verbose
            )
            
            # Save raw predictions
            timestamp = format_timestamp()
            result_data = {
                "timestamp": datetime.now().isoformat(),
                "model_id": args.single_model,
                "model_name": model_name,
                "prompt_id": args.single_prompt,
                "prompt_text": prompt_text,
                "n_samples": len(sampled_images),
                "run_num": run_num,
                "random_seed": current_seed,
                "predictions": predictions
            }
            
            output_file = f"results/raw_predictions/{args.single_model}_{args.single_prompt}_run{run_num}_{timestamp}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result_data, f, indent=2, ensure_ascii=False)
            
            print(f"  ✓ Raw results saved: {output_file}")
            
            # Calculate metrics
            tp, tn, fp, fn = calculate_metrics(predictions, sampled_images)
            
            # Construct CSV row
            csv_row = {
                'true_positive': tp,
                'true_negative': tn,
                'false_positive': fp,
                'false_negative': fn,
                'model_id': args.single_model,
                'prompt_id': args.single_prompt,
                'run_num': run_num,
                'samples': len(sampled_images),
                'timestamp': datetime.now().isoformat()
            }
            
            write_to_csv(csv_path, csv_row)
            
        except Exception as e:
            print(f"  ❌ Run {run_num} failed: {str(e)}")
            # Log failure with -1 metrics
            csv_row = {
                'true_positive': -1,
                'true_negative': -1,
                'false_positive': -1,
                'false_negative': -1,
                'model_id': args.single_model,
                'prompt_id': args.single_prompt,
                'run_num': run_num,
                'samples': len(sampled_images) if 'sampled_images' in locals() else args.samples,
                'timestamp': datetime.now().isoformat()
            }
            write_to_csv(csv_path, csv_row)
            continue
    
    # Summary
    print("\n" + "=" * 80)
    print("Repeated Test Summary")
    print("=" * 80)
    print(f"Total Runs: {args.repeat}")
    print(f"Results recorded in: {csv_path}")
    print(f"Raw data saved in: results/raw_predictions/")
    print("=" * 80 + "\n")


def run_test(args):
    """Run standard multi-model multi-prompt tests"""
    print("\n" + "=" * 80)
    print("VLM Solar Panel Dust Detection Test")
    print("=" * 80)
    
    # 1. Load Dataset
    print("\nStep 1/4: Loading dataset...")
    processor = DatasetProcessor(
        images_dir=args.images_dir,
        index_file=args.index_file
    )
    
    try:
        processor.load_index()
    except FileNotFoundError:
        print("  Dataset index not found, building...")
        processor.build_index()
    
    # Stats
    stats = processor.get_statistics()
    print(f"  Dataset Stats: Total {stats['total']} (Clean: {stats['clean']}, Dirty: {stats['dirty']})")
    
    # 2. Prepare test set
    if args.all_images:
        sampled_images = processor.dataset_index['images']
        print(f"\nStep 2/4: Using all images for testing (n={len(sampled_images)})...")
    else:
        print(f"\nStep 2/4: Sampling test set (n={args.samples})...")
        sampled_images = processor.sample_dataset(
            n_samples=args.samples,
            balanced=True,
            random_seed=args.seed
        )
    
    # 3. Prepare Config
    print("\nStep 3/4: Preparing test configuration...")
    
    # Parse models
    if args.models:
        model_ids = [m.strip() for m in args.models.split(',')]
    else:
        config = load_model_config()
        model_ids = config['default_models']
    
    # Parse prompts
    if args.prompts:
        prompt_ids = [p.strip() for p in args.prompts.split(',')]
    else:
        prompt_ids = ['basic']
    
    # Load model map
    model_config = load_model_config()
    model_map = {m['id']: m['name'] for m in model_config['models']}
    
    print(f"  Target Models: {', '.join(model_ids)}")
    print(f"  Target Prompts: {', '.join(prompt_ids)}")
    print(f"  Total Combinations: {len(model_ids)} × {len(prompt_ids)} = {len(model_ids) * len(prompt_ids)}")
    
    # 4. Run Test
    print("\nStep 4/4: Starting tests...")
    client = VLMClient()
    ensure_dir('results/raw_predictions')
    
    test_results = []
    total_tests = len(model_ids) * len(prompt_ids)
    current_test = 0
    
    for model_id in model_ids:
        for prompt_id in prompt_ids:
            current_test += 1
            print("\n" + "-" * 80)
            print(f"Test {current_test}/{total_tests}: Model={model_id}, Prompt={prompt_id}")
            print("-" * 80)
            
            model_name = model_map.get(model_id)
            if model_name is None:
                print(f"  Error: Model {model_id} not found")
                continue
            
            try:
                prompt_text = get_prompt(prompt_id)
            except ValueError as e:
                print(f"  Error: {e}")
                continue
            
            # Batch query
            predictions = client.batch_query(
                image_list=sampled_images,
                prompt=prompt_text,
                model_name=model_name,
                verbose=args.verbose
            )
            
            # Save raw results
            timestamp = format_timestamp()
            result_data = {
                "timestamp": datetime.now().isoformat(),
                "model_id": model_id,
                "model_name": model_name,
                "prompt_id": prompt_id,
                "prompt_text": prompt_text,
                "n_samples": len(sampled_images),
                "random_seed": args.seed,
                "predictions": predictions
            }
            
            output_file = f"results/raw_predictions/{model_id}_{prompt_id}_{timestamp}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result_data, f, indent=2, ensure_ascii=False)
            
            print(f"\n✓ Raw predictions saved: {output_file}")
            
            # Real-time metric calculation and CSV update
            tp, tn, fp, fn = calculate_metrics(predictions, sampled_images)
            
            csv_row = {
                'true_positive': tp,
                'true_negative': tn,
                'false_positive': fp,
                'false_negative': fn,
                'model_id': model_id,
                'prompt_id': prompt_id,
                'run_num': 1,
                'samples': len(sampled_images),
                'timestamp': datetime.now().isoformat()
            }
            
            csv_path = "vlm_experiment_results.csv"
            write_to_csv(csv_path, csv_row)

            test_results.append({
                "model_id": model_id,
                "prompt_id": prompt_id,
                "output_file": output_file,
                "success_count": sum(1 for p in predictions if p['success'])
            })
    
    # Summary
    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)
    print(f"\nCompleted: {len(test_results)}/{total_tests}")
    for result in test_results:
        print(f"  [{result['model_id']}] x [{result['prompt_id']}]: "
              f"{result['success_count']}/{len(sampled_images)} Success")
    
    print(f"\nRaw results stored in: results/raw_predictions/")
    print(f"Use 'python analyze_results.py' to analyze metrics.")
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='VLM Solar Panel Dust Detection Testing Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Subcommands
    parser.add_argument('--build-index', action='store_true',
                        help='Build dataset index')
    parser.add_argument('--test', action='store_true',
                        help='Run multi-model multi-prompt test (original functionality)')
    parser.add_argument('--repeat-test', action='store_true',
                        help='Run single-model single-prompt repeated test')
    parser.add_argument('--list-models', action='store_true',
                        help='List available models')
    parser.add_argument('--list-prompts', action='store_true',
                        help='List available prompts')
    
    # General parameters
    parser.add_argument('--samples', type=int, default=10,
                        help='Number of samples per test (default: 10)')
    parser.add_argument('--all-images', action='store_true',
                        help='Test all images (ignores --samples)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--verbose', action='store_true',
                        help='Display detailed output')
    
    # Dataset parameters
    parser.add_argument('--images-dir', type=str,
                        default='dataset/archive/solar_panel_dust_segmentation/images',
                        help='Path to images directory')
    parser.add_argument('--index-file', type=str,
                        default='data/dataset_index.json',
                        help='Path to dataset index file')
    parser.add_argument('--force-rebuild', action='store_true',
                        help='Force rebuild index')
    
    # Multi-model parameters
    parser.add_argument('--models', type=str,
                        help='Comma-separated model IDs (e.g., gpt-4o-mini,claude-3-haiku)')
    parser.add_argument('--prompts', type=str,
                        help='Comma-separated prompt IDs (e.g., basic,detailed)')
    
    # Repeated test parameters
    parser.add_argument('--repeat', type=int, default=50,
                        help='Number of repetitions (default: 50, recommended 50-100 for violin plots)')
    parser.add_argument('--single-model', type=str,
                        help='Model ID for repeated test (e.g., gpt-4o-mini)')
    parser.add_argument('--single-prompt', type=str,
                        help='Prompt ID for repeated test (e.g., basic)')
    
    args = parser.parse_args()
    
    # Command execution
    if args.list_models:
        list_available_models()
    elif args.list_prompts:
        list_prompts()
    elif args.build_index:
        build_dataset_index(args)
    elif args.repeat_test:
        # Validate parameters
        if not args.single_model or not args.single_prompt:
            print("❌ Repeated tests require both --single-model and --single-prompt!")
            parser.print_help()
            return
        run_repeated_test(args)
    elif args.test:
        run_test(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()