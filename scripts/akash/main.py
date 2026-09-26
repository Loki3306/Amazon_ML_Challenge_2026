import os
import sys
import argparse
import subprocess

# Ensure src is in python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from inference import run_test_inference


def auto_detect_test_dir():
    candidates = ['dataset/test', 'student_resource/dataset/test']
    for c in candidates:
        if os.path.isdir(c):
            return c
    return candidates[0]


def main():
    parser = argparse.ArgumentParser(description='ML Challenge 2026: Business Entity Resolution Pipeline')
    parser.add_argument('--test-dir', default=auto_detect_test_dir(),
                        help='Path to test dataset directory containing test_source1/2/3.tsv')
    parser.add_argument('--model-path', default='models/final_entity_matcher.joblib',
                        help='Path to saved trained model')
    parser.add_argument('--meta-path', default='models/model_metadata.json',
                        help='Path to saved model metadata')
    parser.add_argument('--output-dir', default='output',
                        help='Directory where matching_results.tsv and candidate_pairs.tsv will be saved')
    parser.add_argument('--batch-size', type=int, default=10000,
                        help='Inference batch size')
    parser.add_argument('--top-k', type=int, default=20,
                        help='Number of candidates per S1 entity')
    parser.add_argument('--validate', action='store_true', default=True,
                        help='Run validate_submission.py on generated outputs')

    args = parser.parse_args()

    # Run inference
    report = run_test_inference(
        test_dir=args.test_dir,
        model_path=args.model_path,
        meta_path=args.meta_path,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        top_k=args.top_k
    )

    # Validate output
    if args.validate:
        validator_script = os.path.join('utils', 'validate_submission.py')
        if not os.path.isfile(validator_script):
            validator_script = os.path.join('student_resource', 'utils', 'validate_submission.py')
        matching_file = os.path.join(args.output_dir, 'matching_results.tsv')
        candidate_file = os.path.join(args.output_dir, 'candidate_pairs.tsv')

        if os.path.isfile(validator_script):
            print('\n' + '=' * 60)
            print('RUNNING OFFICIAL SUBMISSION VALIDATOR')
            print('=' * 60)
            cmd = [
                sys.executable, validator_script,
                '--matching', matching_file,
                '--candidate', candidate_file,
                '--test-dir', args.test_dir
            ]
            print(f"Executing: {' '.join(cmd)}")
            res = subprocess.run(cmd, capture_output=True, text=True)
            print(res.stdout)
            if res.stderr:
                print(res.stderr)
            if res.returncode == 0:
                print('VALIDATION STATUS: PASS')
            else:
                print('VALIDATION STATUS: FAIL')
                sys.exit(1)


if __name__ == '__main__':
    main()
