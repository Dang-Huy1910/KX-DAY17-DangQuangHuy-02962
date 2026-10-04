# Phân tích kết quả Day 17: Memory Systems

Benchmark chạy offline (không cần API key) trên cùng `data/conversations.json` và `data/advanced_long_context.json`.

## Kết quả

### Standard Benchmark (10 hội thoại, user `dungct`)

| Agent    | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
|----------|-------------------|-------------------------|----------------------|------------------|-----------------------|-------------|
| Baseline | 7500              | 32898                   | 0.00                 | 0.20             | 0                     | 0           |
| Advanced | 4844              | 44128                   | 1.00                 | 0.90             | 646                   | 1           |

### Long-Context Stress Benchmark (16 lượt rất dài, user `dungct_stress`)

| Agent    | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
|----------|-------------------|-------------------------|----------------------|------------------|-----------------------|-------------|
| Baseline | 1725              | 32785                   | 0.00                 | 0.20             | 0                     | 0           |
| Advanced | 896               | 9857                    | 1.00                 | 1.00             | 429                   | 12          |

## Câu chuyện hệ thống

1. **Baseline không nhớ dài hạn.** Mỗi `thread_id` là một session độc lập. Câu hỏi recall được hỏi ở thread mới nên recall = 0.
2. **Advanced thêm `User.md` nên recall tăng.** Fact ổn định (tên, nơi ở, nghề, style, đồ uống, pet) được upsert qua nhiều phiên. Correction ("Đà Nẵng → Huế", "backend → MLOps") ghi đè fact cũ.
3. **Hội thoại dài làm prompt cost tăng mạnh.** Baseline gửi lại toàn bộ lịch sử mỗi lượt, nên `Prompt tokens processed` tăng gần như tích lũy theo độ dài thread.
4. **Compact memory kéo chi phí ngữ cảnh xuống.** Stress test kích hoạt 12 lần compact. Advanced còn khoảng 30% prompt tokens so với Baseline (9857 vs 32785) nhưng vẫn recall 1.00.
5. **Hệ thống mạnh hơn nhưng phức tạp hơn.** Phải có guardrail: không ghi fact từ câu hỏi, bỏ nhiễu (Hà Nội chỉ là nơi họp, product manager chỉ là câu đùa), và chấp nhận file `User.md` tăng dần.

## Vì sao Advanced tốn hơn ở hội thoại ngắn?

Ở Standard, compact gần như chưa chạy (1 lần). Mỗi lượt Advanced vẫn nhét `User.md` + summary + cửa sổ message gần nhất vào prompt, nên `Prompt tokens processed` cao hơn Baseline (44128 vs 32898). Đây là phí cố định của persistent memory: đáng trả khi cần nhớ xuyên session, chưa đáng nếu hội thoại luôn ngắn và không cần recall.

## Vì sao compact tối ưu chủ yếu `Prompt tokens processed`?

Compact không làm câu trả lời ngắn đi một cách kỳ diệu. Nó thay lịch sử cũ bằng một summary bị chặn độ dài và chỉ giữ `compact_keep_messages` lượt gần nhất. Cái giảm là **ngữ cảnh mang theo mỗi lượt**, tức cột prompt tokens. `Agent tokens only` phản ánh độ dài câu trả lời; `Memory growth` phản ánh file hồ sơ, không phải transcript.

## Rủi ro memory file

- Ghi nhầm correction (giữ cả Huế lẫn Đà Nẵng) sẽ làm bẩn recall.
- Ghi mọi sở thích tạm thời sẽ làm `User.md` phình mà không giúp prompt.
- Confidence threshold và typed fields giảm rủi ro, nhưng extractor heuristic vẫn có thể bỏ sót fact viết theo cách lạ.

## Bonus đã làm

- **Confidence threshold:** không ghi fact từ câu hỏi / câu đùa / chỗ họp tạm.
- **Entity extraction:** field có cấu trúc (`name`, `location`, `profession`, `style`, ...).
- **Conflict handling:** `upsert_fact` thay dòng cũ, không giữ hai giá trị mâu thuẫn.
- **Memory decay:** fact quá cũ bị hạ ưu tiên khi lắp prompt (`ranked_facts`).
