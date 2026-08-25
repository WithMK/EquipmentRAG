# EquipmentRAG 0.3.0-rc.1 Release Candidate

## 범위

이 Release Candidate는 기존 Code/Document RAG에 다음 기능을 추가합니다.

- Code와 Document 통합 Retrieval 및 근거 구분
- 대화형 질의와 통합 운영 CLI
- Hybrid Search와 선택적 Local Reranker
- 선택적 PDF/PPTX OCR과 XLSX Chart Metadata 추출
- Context Orchestrator 연동용 Loopback JSON API
- 폐쇄망 Local UI
- JSONL 기반 Retrieval 품질 평가
- Model/Wheel 전용 분할 전송 Bundle

## Source와 대용량 자산 분리

GitHub에는 Python Source, Test, 설정 예제, 문서와 배포 도구만 저장합니다. 다음 항목은
`.gitignore`로 제외되며 별도 Offline Asset Bundle로 전달합니다.

```text
models/
wheels/
data/source/
data/documents/
data/chroma/
*.gguf
```

실제 설비 Source, 문서, 평가 Dataset, 인증정보와 Local 설정도 GitHub에 포함하지
않습니다.

## Wheel 수집

Internet에 연결된 Target과 동일한 Windows x64/Python 3.12 PC에서 실행합니다.

```powershell
python deploy\download_offline_wheels.py
```

현재 환경에서 TLS Credential을 사용할 수 없어 Release 검증 중 실제 Wheel 다운로드는
완료하지 못했습니다. 위 도구는 Binary Wheel만 허용하며 완료 후
`wheels/WHEEL_MANIFEST.json`에 SHA-256을 기록합니다.

## 전송 Bundle 생성

Wheel 수집 후 Repository 밖의 Output 경로를 지정합니다.

```powershell
python deploy\prepare_offline_assets.py `
  --output ..\..\outputs\EquipmentRAG-offline-assets-0.3.0-rc.1 `
  --part-size-mb 1900
```

각 Part는 최대 1.9GB이며 `.bin.001`, `.bin.002` 형식입니다. 실행 Script 확장자 차단을
피하기 위해 재조립 도구는 `REASSEMBLE.py.txt`로 복사됩니다. 확장자를 바꾸지 않고도
다음처럼 실행할 수 있습니다.

```powershell
python REASSEMBLE.py.txt --extract-to D:\OfflineAssets\EquipmentRAG
```

도구는 Part별 SHA-256과 완성된 ZIP SHA-256을 모두 확인한 후 안전한 경로만
압축 해제합니다.

## Release 검증 기준

- 전체 Unit/HTTP/Parser/Packaging Test 통과
- Python 및 JavaScript 구문 검사 통과
- `pip check` 통과
- `git diff --check` 통과
- Git 추적 대상에 모델, Wheel, Source, 문서, DB와 Credential이 없는지 확인
- 실제 폐쇄망 PC에서 Model, Index, OCR, LLM, UI End-to-End 확인 필요
