
## Memory efficient LLM


- OOM에러 해결 방법
  - Langchain에서 VLLM 사용
  - 체인으로 나눈다
- vLLM qustndp GPU Utilization 변수가 있는데, 0~1에서 (먼저 할당되는 tensor의 양을 줄일 수 있다)
- 40GB Vram이 있다고 할 때,
- 7B 모델이면 11gb 이상 안 나온다
- 체인으로 엮어서 최초에 한번만

## Flash attention

- KV cache
