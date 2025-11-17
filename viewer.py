#!/usr/bin/env python3
"""
Web-based data viewer for datasets with messages.
"""

import argparse
import json
from flask import Flask, render_template, request, jsonify
from datasets import load_dataset
import os

app = Flask(__name__)

# Global variables to store dataset
current_dataset = None
dataset_path = None
available_datasets = []  # List of available dataset paths
sample_size = None  # Number of rows to sample from datasets


@app.route('/')
def index():
    """Main page with data viewer."""
    return render_template('index.html',
                         dataset_path=dataset_path,
                         available_datasets=available_datasets)


@app.route('/api/available_datasets')
def get_available_datasets():
    """Get list of available datasets."""
    return jsonify({
        'datasets': available_datasets,
        'current': dataset_path
    })


@app.route('/api/load_dataset', methods=['POST'])
def load_dataset_endpoint():
    """Load a specific dataset."""
    global current_dataset, dataset_path, sample_size

    data = request.get_json()
    requested_path = data.get('path')

    if not requested_path:
        return jsonify({'error': 'No path provided'}), 400

    if requested_path not in available_datasets:
        return jsonify({'error': f'Dataset path not in available list: {requested_path}'}), 400

    try:
        print(f"Loading dataset from: {requested_path}")
        current_dataset = load_dataset(requested_path, split="train")
        if sample_size is not None and len(current_dataset) > sample_size:
            current_dataset = current_dataset.take(sample_size)
            print(f"  Sampled {sample_size} rows from dataset")
        dataset_path = requested_path
        print(f"✓ Loaded dataset with {len(current_dataset)} rows")

        return jsonify({
            'success': True,
            'num_rows': len(current_dataset),
            'path': dataset_path
        })
    except Exception as e:
        print(f"✗ Failed to load dataset: {e}")
        return jsonify({'error': f'Failed to load dataset: {str(e)}'}), 500


@app.route('/api/dataset_info')
def dataset_info():
    """Get information about the current dataset."""
    if current_dataset is None:
        return jsonify({'error': 'No dataset loaded'}), 400

    return jsonify({
        'num_rows': len(current_dataset),
        'path': dataset_path
    })


@app.route('/api/row/<int:row_idx>')
def get_row(row_idx):
    """Get a specific row from the dataset."""
    if current_dataset is None:
        return jsonify({'error': 'No dataset loaded'}), 400

    if row_idx < 0 or row_idx >= len(current_dataset):
        return jsonify({'error': f'Row index {row_idx} out of range [0, {len(current_dataset)-1}]'}), 400

    try:
        row_data = current_dataset[row_idx]
        messages = json.loads(row_data['messages'])

        # Extract the fields for each message
        formatted_messages = []
        for msg in messages:
            msg_data = {
                'role': msg.get('role', ''),
                'content': msg.get('content', ''),
                'reasoning_content': msg.get('reasoning_content', '')
            }
            msg_data['original_content'] = msg.get('original_content', '')
            msg_data['original_reasoning_content'] = msg.get('original_reasoning_content', '')
            formatted_messages.append(msg_data)

            tool_calls = msg.get('tool_calls', None)
            if tool_calls is not None:
                tool_call_display_data = {
                    'role': 'tool_calls', # New role as requested
                    'content': json.dumps(tool_calls, indent=2, ensure_ascii=False), # Format as readable JSON string
                    'reasoning_content': '', # No reasoning for tool_calls pseudo-row
                    'original_content': '',
                    'original_reasoning_content': ''
                }
                formatted_messages.append(tool_call_display_data)


        return jsonify({
            'row_idx': row_idx,
            'messages': formatted_messages,
            'total_rows': len(current_dataset)
        })
    except Exception as e:
        return jsonify({'error': f'Error processing row {row_idx}: {str(e)}'}), 500


@app.route('/api/search')
def search():
    """Search through the dataset."""
    if current_dataset is None:
        return jsonify({'error': 'No dataset loaded'}), 400

    query = request.args.get('q', '').lower()
    field = request.args.get('field', 'content')  # role, content, or reasoning_content

    if not query:
        return jsonify({'results': []})

    results = []
    for idx in range(len(current_dataset)):
        try:
            row_data = current_dataset[idx]
            messages = json.loads(row_data['messages'])

            for msg_idx, msg in enumerate(messages):
                field_value = str(msg.get(field, '')).lower()
                if query in field_value:
                    results.append({
                        'row_idx': idx,
                        'msg_idx': msg_idx,
                        'role': msg.get('role', ''),
                        'preview': msg.get(field, '')[:100] + ('...' if len(msg.get(field, '')) > 100 else '')
                    })

                    # Limit results to 100 for performance
                    if len(results) >= 100:
                        break
        except Exception:
            continue

        if len(results) >= 100:
            break

    return jsonify({'results': results, 'count': len(results)})

def main():
    parser = argparse.ArgumentParser(description='Web-based dataset viewer')
    parser.add_argument('--data', type=str, nargs='+', required=True,
                       help='Path(s) to dataset(s). Can specify multiple paths.')
    parser.add_argument('--sample', type=int, default=None, help='Number of rows to sample from the dataset (default: None)')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind to (default: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=8000, help='Port to bind to (default: 8000)')

    args = parser.parse_args()

    global current_dataset, dataset_path, available_datasets, sample_size

    # Store all available dataset paths
    available_datasets = args.data
    sample_size = args.sample

    print(f"Available datasets:")
    for i, path in enumerate(available_datasets, 1):
        print(f"  {i}. {path}")

    # Load the first dataset by default
    print(f"\nLoading first dataset: {available_datasets[0]}")
    if sample_size is not None:
        print(f"  Will sample {sample_size} rows")

    try:
        current_dataset = load_dataset(available_datasets[0], split="train")
        if sample_size is not None:
            current_dataset = current_dataset.take(sample_size)
            print(f"  Sampled {sample_size} rows from dataset")
        dataset_path = available_datasets[0]
        print(f"✓ Loaded dataset with {len(current_dataset)} rows")
    except Exception as e:
        print(f"✗ Failed to load dataset: {e}")
        print("Server will start but no dataset is loaded. You can select one from the UI.")
        current_dataset = None
        dataset_path = None

    print(f"\n🚀 Starting web server at http://{args.host}:{args.port}")
    print(f"   Open your browser and navigate to the URL above")
    print(f"   Press Ctrl+C to stop the server\n")

    app.run(host=args.host, port=args.port, debug=False)


if __name__ == '__main__':
    main()
