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
    global current_dataset, dataset_path

    data = request.get_json()
    requested_path = data.get('path')
    split = data.get('split', 'train')

    if not requested_path:
        return jsonify({'error': 'No path provided'}), 400

    if requested_path not in available_datasets:
        return jsonify({'error': f'Dataset path not in available list: {requested_path}'}), 400

    try:
        print(f"Loading dataset from: {requested_path}")
        current_dataset = load_dataset(requested_path, split=split)
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


def create_html_template():
    """Create the HTML template for the viewer."""
    template_dir = os.path.join(os.path.dirname(__file__), 'templates')
    os.makedirs(template_dir, exist_ok=True)

    html_content = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Dataset Viewer</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }

        .container {
            max-width: 95%;
            margin: 0 auto;
            background: white;
            border-radius: 12px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }

        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
        }

        .header h1 {
            font-size: 2em;
            margin-bottom: 10px;
        }

        .header .dataset-info {
            opacity: 0.9;
            font-size: 0.95em;
        }

        .dataset-selector {
            margin-top: 15px;
            padding: 15px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 8px;
        }

        .dataset-selector h3 {
            font-size: 1em;
            margin-bottom: 10px;
            opacity: 0.9;
        }

        .dataset-buttons {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
        }

        .dataset-btn {
            padding: 8px 16px;
            background: rgba(255, 255, 255, 0.2);
            color: white;
            border: 2px solid rgba(255, 255, 255, 0.3);
            border-radius: 6px;
            cursor: pointer;
            transition: all 0.3s;
            font-size: 0.9em;
            word-break: break-all;
            max-width: 400px;
        }

        .dataset-btn:hover {
            background: rgba(255, 255, 255, 0.3);
            border-color: rgba(255, 255, 255, 0.5);
        }

        .dataset-btn.active {
            background: white;
            color: #667eea;
            border-color: white;
            font-weight: 600;
        }

        .dataset-btn.loading {
            opacity: 0.6;
            cursor: wait;
        }

        .controls {
            padding: 20px 30px;
            background: #f8f9fa;
            border-bottom: 1px solid #e0e0e0;
            display: flex;
            gap: 15px;
            align-items: center;
            flex-wrap: wrap;
        }

        .controls input[type="number"] {
            padding: 10px 15px;
            border: 2px solid #ddd;
            border-radius: 6px;
            font-size: 1em;
            width: 150px;
            transition: border-color 0.3s;
        }

        .controls input[type="number"]:focus {
            outline: none;
            border-color: #667eea;
        }

        .controls button {
            padding: 10px 20px;
            background: #667eea;
            color: white;
            border: none;
            border-radius: 6px;
            font-size: 1em;
            cursor: pointer;
            transition: background 0.3s;
        }

        .controls button:hover {
            background: #5568d3;
        }

        .controls button:disabled {
            background: #ccc;
            cursor: not-allowed;
        }

        .controls .search-box {
            flex: 1;
            min-width: 200px;
            position: relative;
        }

        .controls .search-box input {
            width: 100%;
            padding: 10px 15px;
            border: 2px solid #ddd;
            border-radius: 6px;
            font-size: 1em;
        }

        .content {
            padding: 30px;
            overflow-x: auto;
        }

        .row-info {
            margin-bottom: 20px;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 8px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .row-info .label {
            font-weight: 600;
            color: #667eea;
        }

        .messages-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
            table-layout: fixed;
        }

        .messages-table th {
            background: #667eea;
            color: white;
            padding: 15px;
            text-align: left;
            font-weight: 600;
            position: sticky;
            top: 0;
            overflow-wrap: break-word;
            word-break: break-word;
        }

        .messages-table th:first-child {
            width: 100px;
        }

        .messages-table th:not(:first-child) {
            width: 25%;
        }

        .messages-table td {
            padding: 15px;
            border-bottom: 1px solid #e0e0e0;
            vertical-align: top;
            overflow-wrap: break-word;
            word-break: break-word;
        }

        .messages-table tr:hover {
            background: #f8f9fa;
        }

        .messages-table .role-cell {
            width: 100px;
            font-weight: 600;
            color: #667eea;
        }

        .messages-table .content-cell {
            width: 25%;
            white-space: pre-wrap;
            word-wrap: break-word;
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
        }

        .messages-table .reasoning-cell {
            width: 25%;
            white-space: pre-wrap;
            word-wrap: break-word;
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
            color: #555;
        }

        .error {
            padding: 15px;
            background: #fee;
            border: 2px solid #fcc;
            border-radius: 6px;
            color: #c33;
            margin: 20px 0;
        }

        .loading {
            text-align: center;
            padding: 40px;
            color: #667eea;
            font-size: 1.2em;
        }

        .search-results {
            margin-top: 20px;
            max-height: 300px;
            overflow-y: auto;
            border: 2px solid #ddd;
            border-radius: 6px;
        }

        .search-result-item {
            padding: 10px 15px;
            border-bottom: 1px solid #e0e0e0;
            cursor: pointer;
            transition: background 0.2s;
        }

        .search-result-item:hover {
            background: #f0f0f0;
        }

        .search-result-item .row-info-text {
            font-weight: 600;
            color: #667eea;
            margin-bottom: 5px;
        }

        .search-result-item .preview {
            font-size: 0.9em;
            color: #666;
        }

        .empty-state {
            text-align: center;
            padding: 60px 20px;
            color: #999;
        }

        .empty-state h2 {
            margin-bottom: 10px;
            color: #667eea;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 Dataset Viewer</h1>
            <div class="dataset-info">
                <span id="dataset-path">{{ dataset_path or 'No dataset loaded' }}</span>
                <span id="dataset-size"></span>
            </div>
            {% if available_datasets|length > 1 %}
            <div class="dataset-selector">
                <h3>📁 Available Datasets (click to switch):</h3>
                <div class="dataset-buttons" id="dataset-buttons">
                    {% for dataset in available_datasets %}
                    <button class="dataset-btn {% if dataset == dataset_path %}active{% endif %}"
                            data-path="{{ dataset }}"
                            onclick="switchDataset('{{ dataset }}')">
                        {{ dataset }}
                    </button>
                    {% endfor %}
                </div>
            </div>
            {% endif %}
        </div>

        <div class="controls">
            <div>
                <label for="row-input">Row:</label>
                <input type="number" id="row-input" min="1" value="1">
            </div>
            <button id="prev-btn">← Previous</button>
            <button id="next-btn">Next →</button>
            <div class="search-box">
                <input type="text" id="search-input" placeholder="Search in content...">
            </div>
            <button id="search-btn">🔍 Search</button>
        </div>

        <div class="content">
            <div id="row-info" class="row-info" style="display: none;">
                <span class="label">Current Row: <span id="current-row">0</span> / <span id="total-rows">0</span></span>
                <span class="label">Messages: <span id="num-messages">0</span></span>
            </div>

            <div id="search-results" style="display: none;"></div>

            <div id="loading" class="loading" style="display: none;">
                Loading...
            </div>

            <div id="error" class="error" style="display: none;"></div>

            <div id="empty-state" class="empty-state">
                <h2>Loading dataset...</h2>
                <p>Please wait while row 1 is being loaded</p>
            </div>

            <table id="messages-table" class="messages-table" style="display: none;">
                <thead>
                    <tr id="table-header">
                        <th>Role</th>
                        <th>Content</th>
                        <th>Reasoning Content</th>
                    </tr>
                </thead>
                <tbody id="messages-body">
                </tbody>
            </table>
        </div>
    </div>

    <script>
        let currentRow = 0;
        let totalRows = 0;

        // Switch to a different dataset
        async function switchDataset(datasetPath) {
            // Show loading state
            const buttons = document.querySelectorAll('.dataset-btn');
            buttons.forEach(btn => {
                if (btn.dataset.path === datasetPath) {
                    btn.classList.add('loading');
                    btn.disabled = true;
                }
            });

            showLoading();
            hideError();

            try {
                const response = await fetch('/api/load_dataset', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        path: datasetPath,
                        split: 'train'
                    })
                });

                const data = await response.json();

                if (data.error) {
                    showError('Failed to load dataset: ' + data.error);
                    hideLoading();
                    return;
                }

                // Update UI
                document.getElementById('dataset-path').textContent = data.path;
                document.getElementById('dataset-size').textContent = ` (${data.num_rows} rows)`;
                totalRows = data.num_rows;

                // Update active button
                buttons.forEach(btn => {
                    btn.classList.remove('active', 'loading');
                    btn.disabled = false;
                    if (btn.dataset.path === datasetPath) {
                        btn.classList.add('active');
                    }
                });

                // Reset view and load row 1 automatically (index 0)
                currentRow = 0;
                document.getElementById('row-input').value = 1;
                document.getElementById('row-input').max = totalRows;

                hideLoading();

                console.log(`✓ Switched to dataset: ${datasetPath}`);

                // Automatically load row 1 (index 0)
                loadRow(0);
            } catch (error) {
                showError('Error switching dataset: ' + error.message);
                hideLoading();

                // Remove loading state
                buttons.forEach(btn => {
                    btn.classList.remove('loading');
                    btn.disabled = false;
                });
            }
        }

        // Load dataset info
        async function loadDatasetInfo() {
            try {
                const response = await fetch('/api/dataset_info');
                const data = await response.json();

                if (data.error) {
                    showError(data.error);
                    return;
                }

                totalRows = data.num_rows;
                document.getElementById('total-rows').textContent = totalRows;
                document.getElementById('dataset-size').textContent = ` (${totalRows} rows)`;
            } catch (error) {
                showError('Failed to load dataset info: ' + error.message);
            }
        }

        // Load a specific row
        async function loadRow(rowIdx) {
            hideError();
            hideSearchResults();
            showLoading();

            try {
                const response = await fetch(`/api/row/${rowIdx}`);
                const data = await response.json();

                if (data.error) {
                    showError(data.error);
                    hideLoading();
                    return;
                }

                currentRow = data.row_idx;
                totalRows = data.total_rows;

                displayMessages(data.messages);
                updateRowInfo(currentRow, totalRows, data.messages.length);
                updateControls();

                hideLoading();
                document.getElementById('empty-state').style.display = 'none';
            } catch (error) {
                showError('Failed to load row: ' + error.message);
                hideLoading();
            }
        }

        // Display messages in the table
        function displayMessages(messages) {
            const tbody = document.getElementById('messages-body');
            const thead = document.getElementById('table-header');
            tbody.innerHTML = '';

            // Check if original content fields exist
            const hasOriginal = messages.length > 0 &&
                (messages[0].hasOwnProperty('original_content') ||
                 messages[0].hasOwnProperty('original_reasoning_content'));

            // Update table headers dynamically
            if (hasOriginal) {
                thead.innerHTML = `
                    <th>Role</th>
                    <th>Original Content</th>
                    <th>Content</th>
                    <th>Original Reasoning</th>
                    <th>Reasoning</th>
                `;
            } else {
                thead.innerHTML = `
                    <th>Role</th>
                    <th>Content</th>
                    <th>Reasoning Content</th>
                `;
            }

            messages.forEach((msg, idx) => {
                const row = tbody.insertRow();

                const roleCell = row.insertCell();
                roleCell.className = 'role-cell';
                roleCell.textContent = msg.role;

                if (hasOriginal) {
                    // Original content column
                    const originalContentCell = row.insertCell();
                    originalContentCell.className = 'content-cell';
                    originalContentCell.textContent = msg.original_content || '';

                    // Translated content column
                    const contentCell = row.insertCell();
                    contentCell.className = 'content-cell';
                    contentCell.textContent = msg.content;

                    // Original reasoning column
                    const originalReasoningCell = row.insertCell();
                    originalReasoningCell.className = 'reasoning-cell';
                    originalReasoningCell.textContent = msg.original_reasoning_content || '';

                    // Translated reasoning column
                    const reasoningCell = row.insertCell();
                    reasoningCell.className = 'reasoning-cell';
                    reasoningCell.textContent = msg.reasoning_content;
                } else {
                    // Normal mode: just show translated content
                    const contentCell = row.insertCell();
                    contentCell.className = 'content-cell';
                    contentCell.textContent = msg.content;

                    const reasoningCell = row.insertCell();
                    reasoningCell.className = 'reasoning-cell';
                    reasoningCell.textContent = msg.reasoning_content;
                }
            });

            document.getElementById('messages-table').style.display = 'table';
        }

        // Update row info display (convert 0-based to 1-based)
        function updateRowInfo(current, total, numMessages) {
            document.getElementById('current-row').textContent = current + 1;
            document.getElementById('total-rows').textContent = total;
            document.getElementById('num-messages').textContent = numMessages;
            document.getElementById('row-info').style.display = 'flex';
        }

        // Update control buttons state (convert 0-based to 1-based for display)
        function updateControls() {
            document.getElementById('prev-btn').disabled = currentRow <= 0;
            document.getElementById('next-btn').disabled = currentRow >= totalRows - 1;
            document.getElementById('row-input').value = currentRow + 1;
            document.getElementById('row-input').max = totalRows;
        }

        // Show/hide loading state
        function showLoading() {
            document.getElementById('loading').style.display = 'block';
        }

        function hideLoading() {
            document.getElementById('loading').style.display = 'none';
        }

        // Show/hide error message
        function showError(message) {
            const errorDiv = document.getElementById('error');
            errorDiv.textContent = message;
            errorDiv.style.display = 'block';
        }

        function hideError() {
            document.getElementById('error').style.display = 'none';
        }

        // Show/hide search results
        function hideSearchResults() {
            document.getElementById('search-results').style.display = 'none';
        }

        // Search functionality
        async function performSearch() {
            const query = document.getElementById('search-input').value;
            if (!query) {
                hideSearchResults();
                return;
            }

            showLoading();
            hideError();

            try {
                const response = await fetch(`/api/search?q=${encodeURIComponent(query)}&field=content`);
                const data = await response.json();

                if (data.error) {
                    showError(data.error);
                    hideLoading();
                    return;
                }

                displaySearchResults(data.results, data.count);
                hideLoading();
            } catch (error) {
                showError('Search failed: ' + error.message);
                hideLoading();
            }
        }

        // Display search results
        function displaySearchResults(results, count) {
            const resultsDiv = document.getElementById('search-results');

            if (results.length === 0) {
                resultsDiv.innerHTML = '<div style="padding: 20px; text-align: center;">No results found</div>';
            } else {
                let html = `<div style="padding: 10px; background: #f0f0f0; font-weight: 600;">Found ${count} result${count > 1 ? 's' : ''}</div>`;
                results.forEach(result => {
                    html += `
                        <div class="search-result-item" onclick="loadRow(${result.row_idx})">
                            <div class="row-info-text">Row ${result.row_idx + 1}, Message ${result.msg_idx + 1} (${result.role})</div>
                            <div class="preview">${escapeHtml(result.preview)}</div>
                        </div>
                    `;
                });
                resultsDiv.innerHTML = html;
            }

            resultsDiv.style.display = 'block';
        }

        // Escape HTML for display
        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }

        // Event listeners
        document.getElementById('prev-btn').addEventListener('click', () => {
            if (currentRow > 0) {
                loadRow(currentRow - 1);
            }
        });

        document.getElementById('next-btn').addEventListener('click', () => {
            if (currentRow < totalRows - 1) {
                loadRow(currentRow + 1);
            }
        });

        document.getElementById('search-btn').addEventListener('click', performSearch);

        document.getElementById('search-input').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                performSearch();
            }
        });

        document.getElementById('row-input').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                const rowNum = parseInt(document.getElementById('row-input').value);
                // Convert 1-based input to 0-based index
                loadRow(rowNum - 1);
            }
        });

        // Load dataset info on page load
        async function initializePage() {
            await loadDatasetInfo();
            // Automatically load row 1 (index 0) if dataset is loaded
            if (totalRows > 0) {
                loadRow(0);
            }
        }

        initializePage();
    </script>
</body>
</html>'''

    template_path = os.path.join(template_dir, 'index.html')
    with open(template_path, 'w') as f:
        f.write(html_content)

    print(f"✓ Created HTML template at {template_path}")


def main():
    parser = argparse.ArgumentParser(description='Web-based dataset viewer')
    parser.add_argument('--data', type=str, nargs='+', required=True,
                       help='Path(s) to dataset(s). Can specify multiple paths.')
    parser.add_argument('--split', type=str, default='train', help='Dataset split to load (default: train)')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind to (default: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=8000, help='Port to bind to (default: 8000)')

    args = parser.parse_args()

    global current_dataset, dataset_path, available_datasets

    # Store all available dataset paths
    available_datasets = args.data

    print(f"Available datasets:")
    for i, path in enumerate(available_datasets, 1):
        print(f"  {i}. {path}")

    # Load the first dataset by default
    print(f"\nLoading first dataset: {available_datasets[0]}")
    print(f"Split: {args.split}")

    try:
        current_dataset = load_dataset(available_datasets[0], split=args.split)
        dataset_path = available_datasets[0]
        print(f"✓ Loaded dataset with {len(current_dataset)} rows")
    except Exception as e:
        print(f"✗ Failed to load dataset: {e}")
        print("Server will start but no dataset is loaded. You can select one from the UI.")
        current_dataset = None
        dataset_path = None

    # Create HTML template
    create_html_template()

    print(f"\n🚀 Starting web server at http://{args.host}:{args.port}")
    print(f"   Open your browser and navigate to the URL above")
    print(f"   Press Ctrl+C to stop the server\n")

    app.run(host=args.host, port=args.port, debug=False)


if __name__ == '__main__':
    main()

