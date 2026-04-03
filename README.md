# Soft-Actor-Critic

SAC 스켈레톤 코드임ㅇㅇ

automatic temperature adjustment 버전임

```
pip install uv
uv run main.py
```


> [!TIP]
> 코드 스타일 통일화를 위해 editorconfig, black, ruff 등을 설치해뒀으니 귀찮으면 사용 권장

1. editorconfig

[여기](https://marketplace.visualstudio.com/items?itemName=EditorConfig.EditorConfig)에서 vscode extension 다운로드 가능

.editorconfig 작성해 뒀으니 저장할때 알아서 스타일링 해줄거임

들여쓰기같은 간단한 스타일 유지하는데 도와줌

2. black

얘도 스타일링 용인데 얘는 저장 후 동작함

좀 더 엄격하고 읽기 쉽게 포매팅해줌

3. ruff

얘는 린터 + 최적화 해주는애임

import 순서 규율같은거까지 엄격하게 잡아줌

### 생각하기 싫으면 그냥 이거 실행하면 됨

```
./style.sh
```
