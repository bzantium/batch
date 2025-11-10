# bzantium/batch/batch-d8b0200624b4621212732e896f3864de7f3fe956/view_data.py

import argparse
import json
...
from typing import Optional, List # [추가]

app = Flask(__name__)
...

# ... (API 엔드포인트 @app.route(...) 등은 변경 없음) ...

def create_html_template():
    ...

# [수정] main -> run_view_logic로 변경
# [수정] args_list를 인자로 받아, None이 아니면 sys.argv 대신 사용
def run_view_logic(args_list: Optional[List[str]] = None):
    """
    웹 기반 데이터 뷰어를 실행합니다.
    (main.py에서 호출 가능하도록 리팩토링됨)
    """
    parser = argparse.ArgumentParser(description='Web-based dataset viewer')
    parser.add_argument('--data', type=str, nargs='+', required=True,
                       help='Path(s) to dataset(s). Can specify multiple paths.')
    parser.add_argument('--split', type=str, default='train', help='Dataset split to load (default: train)')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind to (default: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=8000, help='Port to bind to (default: 8000)')

    # [수정] args_list가 None이면 sys.argv[1:]를, 아니면 args_list를 파싱
    args = parser.parse_args(args_list)

    global current_dataset, dataset_path, available_datasets

    # ... (기존 main 함수의 나머지 로직) ...

    available_datasets = args.data
    ...

    create_html_template()

    print(f"\n🚀 Starting web server at http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == '__main__':
    # [수정] run_view_logic() 호출 (인자 없이 호출하여 sys.argv 사용)
    run_view_logic()