# Lab 21 — Evaluation Report

**Họ tên**: Ngô Ngọc Quyên  **MSSV**: 2A202601928  **Ngày**: 21/08/2026
**Tier**: `T4`  **Base model**: `unsloth/Qwen3.5-4B`  **GPU thực tế**: `T4 16GB`

Mọi số liệu dưới đây khớp với các tệp trong `results/`. Toàn bộ đánh giá dùng đủ 50 mẫu target và 15 mẫu regression, không giới hạn `EVAL_LIMIT`.

## 1. Setup

| | |
|---|---|
| Dataset | 250 ticket CSKH → JSON triage |
| Train / val | 225 / 25 (seed 42) |
| `max_length` | 1024 — p95 đo được là 98 token |
| `MASK_MODE` | `assistant-only` |
| Epochs / max_steps | 1 / 30 |

Template có giữ khối `<think>` không? **Có** — `template_check.json` xác nhận reasoning được bảo toàn, nên có thể huấn luyện trace an toàn. Token dài nhất chỉ 101 nhưng giữ `max_length=1024` theo cấu hình T4 để không cắt các ticket thực tế dài hơn bộ seed.

## 2. Mask proof (NB1)

| | |
|---|---|
| `supervised_fraction` | 0.4149 |
| Câu trả lời nằm trong loss | true |
| Câu hỏi KHÔNG nằm trong loss | true |

Đoạn được tính loss bắt đầu tại phần assistant, sau trace `<think>`:

```
</think>
{"intent": "doi_tra", "urgency": "trung_binh",
 "product": "balo laptop", "sentiment": "trung_tinh"}
<|im_end|>
```

Như vậy model học đầu ra JSON/trace thay vì học lặp lại system prompt hay ticket của khách.

## 3. Ba baseline (NB2 — đo trước khi train)

| Run | target | regression | format | latency (ms) |
|---|---:|---:|---:|---:|
| (a) base + naive prompt | 0.000 | 0.758 | 0.000 | 3210.4 |
| (b) base + optimized prompt | 0.765 | 0.758 | 1.000 | 1041.3 |
| (c) LoRA fine-tune | 0.965 | 0.589 | 1.000 | 1448.9 |

(b) thực sự mạnh hơn (a): target tăng từ 0.000 lên 0.765, JSON hợp lệ hoàn toàn và nhanh hơn đáng kể. Tôi không sửa `OPTIMIZED_PROMPT`; hash `719e74d3b6232053` khớp prompt gốc. Đây là baseline công bằng và đủ mạnh để so sánh với fine-tune.

## 4. Giải phẫu cấu hình sai (NB4)

| Run | vị trí | r | trainable | LR | train loss | target | thời gian (s) | VRAM GB |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `correct` | all text-linear | 16 | 32,464,896 | 1e-4 | 0.6292 | 0.965 | 979.6 | 12.01 |
| `attn_only` | q,v | 283 (matched) | 32,456,704 | 1e-4 | 0.5379 | 0.970 | 802.2 | 12.02 |
| `wrong_lr` | all text-linear | 16 | 32,464,896 | 1e-5 | 1.5704 | 0.000 | 939.2 | 12.01 |
| `qlora` | all text-linear | 16 | 32,464,896 | 1e-4 | 0.7058 | 0.940 | 1003.8 | 7.09 |

**4.1.** `attn_only` được tăng rank lên 283 để số tham số trainable gần bằng `correct` (chênh chưa đến 0.1%). Trên target nó nhỉnh hơn rất ít, 0.970 so với 0.965; vì vậy trong phép đo này hai cấu hình gần như hòa, không thể kết luận placement text-linear luôn tốt hơn. Thứ tự đó cũng cùng chiều train loss: `attn_only` có loss 0.5379 thấp hơn 0.6292. Bài học là không được thay rank và vị trí adapter cùng lúc: rank đủ lớn có thể bù một phần cho placement, nên cần matched-rank để phép đối chiếu có ý nghĩa.

**4.2.** `wrong_lr` chỉ đổi LR từ 1e-4 xuống 1e-5 nhưng loss cuối tăng lên 1.5704, trong khi cấu hình đúng là 0.6292. Nó chưa học đủ trong cùng 30 bước, đồng thời target và format đều bằng 0. Nếu chỉ thấy loss cao mà không biết LR, tôi có thể kết luận nhầm rằng LoRA placement hoặc dữ liệu hỏng. Thí nghiệm cô lập một biến cho thấy nguyên nhân là learning rate mang thang đo của full fine-tuning sang LoRA.

**4.3.** QLoRA giảm VRAM từ 12.01 xuống 7.09 GB, tiết kiệm 4.92 GB (khoảng 41%). Đổi lại target giảm 0.025, loss tăng 0.0766 và latency tăng từ 1448.9 lên 1811.3 ms. Các số đo vì thế ủng hộ khuyến nghị ưu tiên fp16 LoRA cho dòng model này khi 12 GB vẫn vừa; QLoRA vẫn là lựa chọn hợp lý khi bộ nhớ là ràng buộc chính. Kết luận là trade-off thực nghiệm, không phải cấm tuyệt đối QLoRA.

## 5. Phán quyết (NB5)

**Kết quả cổng hồi quy**: **FAILED**
`target Δ = +0.200` · `regression Δ = -0.169` · `valid_trace_rate = 0.00`

Fine-tune thắng rõ trên target: 0.965 so với baseline prompt mạnh (b) là 0.765. Tuy nhiên regression giảm từ 0.758 xuống 0.589, tức mất 0.169, lớn hơn rất nhiều ngưỡng cho phép 0.020. Vì vậy verdict FAILED là kết quả đúng và không nên nới gate để đổi thành PASSED. Dữ liệu train seed tập trung vào nhiệm vụ triage nên adapter đã chuyên biệt hóa tốt, nhưng cái giá là suy giảm năng lực tổng quát của mô hình trên tập regression. `valid_trace_rate` bằng 0 vì output được chấm là JSON trực tiếp, không được dùng làm lý do bỏ qua regression gate. Bản này cho thấy target score đơn lẻ không đủ để quyết định deploy. Cách sửa tiếp theo là trộn 1–5% replay data đại diện năng lực gốc, giữ prompt/bộ đánh giá cố định rồi chạy lại toàn bộ NB2–NB5.

## 6. Định tính — có cả ca thua

| # | Ticket (rút gọn) | Nhãn đúng | (b) prompt | (c) fine-tune | Nhận xét |
|---|---|---|---|---|---|
| 1 | Chuột không dây, muốn trả lại, gấp | doi_tra/cao | hoan_tien/cao | doi_tra/cao | FT thắng: sửa intent |
| 2 | Ốp lưng, hoàn tiền, sớm | hoan_tien/trung_binh | hoan_tien/cao | hoan_tien/trung_binh | FT thắng: sửa urgency |
| 3 | Bình giữ nhiệt, chưa thấy tiền | hoan_tien/thap | hoan_tien/trung_binh | hoan_tien/trung_binh | FT thua: urgency |
| 4 | Nồi chiên thiếu phụ kiện | san_pham_loi/thap | hoan_tien/cao | san_pham_loi/trung_binh | FT thua: chỉ còn sai urgency |
| 5 | Áo khoác gió bị lỗi | san_pham_loi/thap | san_pham_loi/trung_binh | san_pham_loi/trung_binh | FT thua: urgency |

Mẫu chung ở các ca FT thua là câu “Khi nào tiện” bị model hiểu thành urgency trung bình, trong khi nhãn đặt là thấp. FT cải thiện intent/product/sentiment nhưng chưa học được quy ước nhãn urgency tinh tế này; đây là lỗi dữ liệu/định nghĩa nhãn cần bổ sung ví dụ đối chứng.

## 7. Kết luận & điều tôi học được

**Kết luận.** Tôi chưa deploy bản fine-tune này dù nó tăng target score 0.200. Gate regression đã phát hiện suy giảm 0.169, chứng tỏ adapter đang tối ưu quá mạnh cho phân phối ticket seed và làm yếu hành vi ngoài phân phối đó. Đòn bẩy quan trọng nhất trong lab không phải chỉ là rank: mask `assistant-only` bảo đảm gradient đi vào đáp án, learning rate quyết định model có học được trong ngân sách bước hay không, còn chất lượng và độ phủ dữ liệu quyết định việc tăng target có đánh đổi general capability. Placement adapter có ảnh hưởng nhưng kết quả matched-rank cho thấy cần đo chứ không nên khẳng định theo trực giác. Tôi sẽ giữ baseline (b), thêm replay data và ví dụ urgency đối chứng, sau đó lặp lại cùng 30 bước và cùng eval set. Chỉ khi target vẫn tốt và regression vượt gate mới có cơ sở deploy. QLoRA là phương án tiết kiệm bộ nhớ đáng cân nhắc, nhưng ở máy T4 đang có đủ VRAM thì fp16 LoRA đem lại target tốt hơn và suy luận nhanh hơn.

**Ba điều tôi học được:**

1. Prompt baseline mạnh có thể đạt 0.765; phải đóng băng nó trước train để không biến prompt yếu thành đối thủ giả.
2. Cùng budget không đủ, các contrast còn phải matched trainable parameters; rank 283 giúp đối chiếu q,v với all-linear công bằng.
3. Target score 0.965 vẫn không đủ để deploy: regression gate đã bắt đúng trường hợp over-specialization.

**Nếu có thêm 2 giờ nữa, tôi sẽ thử:** thêm 1–5% replay data và ví dụ contrast cho “Khi nào tiện”, sau đó rerun NB2–NB5 với cùng seed, prompt và full eval set.
