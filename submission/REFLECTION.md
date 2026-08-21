# Reflection — Lab 21

## 1. Điều gì làm tôi ngạc nhiên nhất?

Prompt tối ưu của base model đã đạt target 0.765, trong khi prompt ngây thơ có target 0.000 và format 0.000. Điều này làm tôi thấy rõ fine-tuning không nên được so với một baseline yếu. Một kết quả khác cũng đáng chú ý là `attn_only` với matched-rank 283 đạt 0.970, gần như hòa và nhỉnh hơn `correct` 0.965; trực giác “gắn adapter vào nhiều lớp hơn chắc chắn tốt hơn” không được dữ liệu này xác nhận.

## 2. Tôi mất nhiều thời gian nhất ở đâu? Có đúng như dự đoán không?

Phần lâu nhất là sinh đánh giá đầy đủ và chạy ba contrast trên T4, không phải bước cài đặt hay kiểm tra mask. Sinh output cho 50 target và 15 regression lặp lại cho nhiều nhóm đo khiến thời gian thực tế dài hơn cảm giác từ 30 bước train. Điều này đúng với ghi chú của lab rằng generation chiếm phần lớn thời gian, nhưng tôi vẫn ban đầu đánh giá thấp chi phí của việc giữ phép so sánh công bằng.

## 3. Niềm tin nào về fine-tuning mà tôi không còn tin nữa?

Tôi không còn xem target accuracy cao là đủ để deploy. Fine-tune tăng target từ 0.765 lên 0.965 nhưng làm regression giảm từ 0.758 xuống 0.589, nên regression gate cho verdict FAILED. Tôi cũng không coi training loss là bảng xếp hạng cuối cùng: nó là tín hiệu tối ưu hóa, còn target/regression mới phản ánh hành vi cần giữ.

## 4. Tôi dùng AI assistant vào việc gì? Chỗ nào nó sai?

Tôi dùng AI assistant để đọc cấu trúc lab, đối chiếu artifact với rubric, tự động hóa kiểm tra và giúp diễn giải số đo thành report. AI không được dùng để bịa kết quả; các số trong report được lấy từ run T4 và kiểm tra lại bằng gatekeeper. Một giới hạn quan trọng là AI dễ diễn giải theo trực giác rằng all-linear sẽ hơn q,v; kết quả matched-rank của lab cho thấy phải ưu tiên số đo hơn suy đoán đó.

## 5. Nếu ngày mai fine-tune cho khách hàng thật, bước đầu tiên là gì?

Tôi sẽ xác định metric, tập regression và tiêu chí rollback trước khi chạm vào dữ liệu train. Sau đó tôi lập baseline prompt đủ mạnh, kiểm tra schema/nhãn và tạo một tập holdout không bị thay đổi sau khi đã nhìn kết quả. Với kết quả hiện tại, bước tiếp theo cụ thể là bổ sung 1–5% replay data và các ví dụ đối chứng cho nhãn urgency rồi chạy lại cùng protocol, thay vì deploy adapter hiện tại.
