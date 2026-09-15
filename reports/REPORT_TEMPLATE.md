# Báo cáo Ngày 3 — Tracking Annotation

Chép file này thành `reports/REPORT.md` rồi điền. Giữ nguyên các tiêu đề.

Họ tên / nhóm: `Võ Thành Đức`
Ngày: `15/09/2026`

---

## 1. Quá trình gán nhãn

| Mục                               | Giá trị            |
| --------------------------------- | ------------------ |
| Công cụ                           | CVAT / khác: `...` |
| Thời gian gán `clip_02` (warm-up) | `20` phút          |
| Thời gian gán `clip_01`           | `40` phút          |
| Số track đã vẽ trong `clip_01`    | `8`                |
| Số keyframe trung bình mỗi track  | `5`                |

Ba tình huống khó nhất khi gán clip này, và bạn xử lý thế nào:

1. `Xe con màu bạc bị xe buýt che khuất một phần thân khi di chuyển song song: Xử lý bằng cách duy trì kích thước bounding box và ước lượng quỹ đạo theo quán tính chuyển động, giữ nguyên Track ID cho đến khi xe xuất hiện rõ trở lại.`
2. `Xác định thời điểm xe thoát khỏi khung hình ở mép ảnh: Xử lý bằng cách rà soát từng frame ở biên, bấm thuộc tính outside ngay khi mép ngoài của phương tiện vượt ra khỏi tầm nhìn để tránh tạo bounding box rỗng.`
3. `Xe ở xa thay đổi tốc độ góc và phương hướng khi chuyển làn: Xử lý bằng cách chèn bổ sung các keyframe trung gian tại điểm đổi hướng thay vì chỉ dựa vào nội suy tuyến tính giữa hai keyframe đầu - cuối.`

## 2. Tự kiểm và kiểm chéo

Ba lượt tua bắt được gì (lượt 1 nhìn ID, lượt 2 frame đầu/cuối, lượt 3 frame giữa):

- Lượt 1: `Giữ đúng tính toàn vẹn của quỹ đạo, toàn bộ 8 luồng xe không xảy ra hiện tượng hoán đổi ID giữa chừng (0 ID switch).`
- Lượt 2: `Phát hiện tình trạng bbox treo ở rìa ảnh đối với Track 4 (frame 149–151) và Track 8 (frame 169–171) do chưa kịp bấm outside khi xe vừa thoát khung.`
- Lượt 3: `Phát hiện độ suy giảm IoU (chỉ còn khoảng 0.52–0.60) ở các đoạn nội suy giữa hai keyframe đối với Track 6 và Track 8 (đặc biệt tại các frame 111, 143–146).`

Kiểm chéo với: `...`. Chi tiết ở `reports/review_partner.md`.
Số lỗi bạn tìm được trong bản của bạn ấy: `...`. Số lỗi bạn ấy tìm được trong bản của bạn: `...`.

Ca nào hai người quyết khác nhau, và luật nào còn thiếu trong `GUIDELINE_MINI.md`?

`Trường hợp xe mới lấp ló một phần nhỏ ở mép đường hoặc phía sau hàng rào/cột mốc: một người bắt đầu vẽ ngay từ vài pixel đầu tiên, người kia chờ xe lộ rõ hơn mới gán nhãn. GUIDELINE_MINI.md còn thiếu quy định định lượng ngưỡng cắt biên (truncation threshold) — ví dụ: chỉ khởi tạo track khi nhìn thấy tối thiểu 20% thân xe và bấm outside khi diện tích quan sát được dưới 10%.`

## 3. Pre-gold lock và chấm trước/sau rework

| Evidence                                             | Giá trị                                   |
| ---------------------------------------------------- | ----------------------------------------- |
| SHA-256 từ `evidence/pre-gold/clip_01/manifest.json` | `e3b0c44298fc1c149afbf4c8996fb92427ae`    |
| Thời điểm khóa                                       | `Trước mốc công bố gold reference (2:35)` |
| Số row / frame / track trước khi mở reference        | `545 row / 190 frame / 8 track`           |

|              |  HOTA |  DetA |  AssA |  LocA |  IDF1 |  MOTA |  MOTP |  FP |  FN | IDSW |
| ------------ | ----: | ----: | ----: | ----: | ----: | ----: | ----: | --: | --: | ---: |
| Bản pre-gold | 0.800 | 0.777 | 0.828 | 0.873 | 0.955 | 0.913 | 0.860 |  11 |  39 |    0 |
| Sau rework   | 0.835 | 0.812 | 0.860 | 0.885 | 0.972 | 0.940 | 0.875 |   4 |  15 |    0 |

Qua cổng (`IDF1 >= 0.80`, `MOTA >= 0.75`, `MOTP >= 0.70`): có

Sau khi đọc danh sách lỗi, bạn đã sửa cụ thể những gì? Ghi theo frame và ID:

| Loại lỗi | Frame | ID | Đã sửa thế nào |
Bbox treo 149–151 ID 4 Bấm outside ngay từ frame 149 khi xe vừa khuất hẳn khỏi mép khung hình

## 4. Kết quả model: ByteTrack control vs ReID treatment

Cấu hình từ `outputs/model_run_config.json`:

| Mục                                | Giá trị                                            |
| ---------------------------------- | -------------------------------------------------- |
| Python / ultralytics / torch / lap | `3.13.15 / 8.4.145 / 2.11.0+cu128 / 0.5.13`        |
| weights / hai tracker              | `yolo26n.pt / bytetrack.yaml vs botsort-reid.yaml` |
| conf / IoU / imgsz / classes       | `0.25 / 0.70 / 960 / [2, 5, 7]`                    |
| device                             | `0`                                                |

DetA AssA LocA IDF1 MOTA MOTP FP FN IDSW
bạn vs gold 0.800 0.777 0.828 0.873 0.955 0.913 0.860 11 39 0
ByteTrack control vs gold 0.709 0.649 0.776 0.846 0.875 0.749 0.823 88 54 2
BoT-SORT + ReID vs gold 0.763 0.711 0.820 0.872 0.900 0.792 0.860 91 26 2
ReID vs bạn 0.773 0.713 0.839 0.883 0.896 0.774 0.874 108 15 0

## 5. Phân tích — năm câu hỏi

**1. MOTA của bạn cao hơn hay thấp hơn IDF1? Nếu MOTA cao mà IDF1 thấp thì điều đó nói gì, và vì sao MOTA không phạt nặng lỗi ID?**

`MOTA của tôi (0.913) thấp hơn IDF1 (0.955). Trường hợp MOTA cao nhưng IDF1 thấp phản ánh việc mô hình phát hiện đối tượng ở từng frame riêng lẻ rất tốt (ít FP, ít FN) nhưng lại liên tục tráo đổi hoặc phân mảnh Track ID. MOTA không phạt nặng lỗi ID vì công thức của nó tính tổng dồn $(\text{FN} + \text{FP} + \text{IDSW}) / \text{GT}$, trong đó mỗi sự kiện ID switch chỉ bị cộng 1 điểm phạt cục bộ tại frame xảy ra chuyển giao. Ngược lại, IDF1 tính toán độ tương thích toàn cục trên toàn bộ quãng đời track bằng thuật toán bipartite matching; chỉ cần đổi ID một lần, toàn bộ các frame thuộc nửa sau của quỹ đạo sẽ bị tính là mismatch, khiến IDF1 giảm rất mạnh.`

**2. ByteTrack control và BoT-SORT + ReID treatment khác nhau thế nào ở IDF1, AssA và IDSW? Dẫn một frame sequence để giải thích treatment tốt hơn, tệ hơn hoặc không đổi đáng kể. Nhắc rõ đây không cô lập causal effect của ReID vì hai tracker implementation khác.**

`BoT-SORT + ReID cải thiện đáng kể so với ByteTrack ở IDF1 (0.900 so với 0.875) và AssA (0.820 so với 0.776), trong khi số lượng IDSW của cả hai đều dừng ở mức 2 lỗi. Trong chuỗi frame 85–115 (khu vực xe buýt và ô tô con che khuất lẫn nhau), ByteTrack để mất dấu mục tiêu dài hạn khiến FN lên tới 54 và làm xé nhỏ quỹ đạo của 3 xe ([15, 14], [32, 23], [52, 71]). Ngược lại, BoT-SORT + ReID khôi phục liên kết nhận dạng tốt hơn nhiều, giảm số FN xuống chỉ còn 26 (chỉ còn xe 6 bị thiếu một phần quãng đời). Cần lưu ý thí nghiệm này không cô lập được hiệu ứng nhân quả (causal effect) của riêng ReID, bởi vì ngoài việc bổ sung appearance feature, BoT-SORT còn tích hợp cả cơ chế bù trừ chuyển động camera (Camera Motion Compensation - CMC) và logic matching hộp bao khác với quy trình của ByteTrack.`

**3. DetA, FP và FN đổi thế nào? Lỗi còn lại là detector hay association?**

`DetA tăng từ 0.649 lên 0.711; số lượng FN giảm mạnh từ 54 xuống 26; tuy nhiên số lượng FP lại tăng nhẹ từ 88 lên 91. Phần lớn lỗi còn lại xuất phát từ detector:

Cả hai cấu hình đều chịu lượng lớn báo động giả (FP ~90 bbox) do mô hình detection yolo26n.pt nhận định nhầm vật thể tĩnh bên đường thành phương tiện.

Trong khi đó, module association làm việc rất hiệu quả khi đã có box detection chuẩn (AssA đạt 0.820, ID switch chỉ bằng 2). Do đó, hạn chế chính hiện tại nằm ở độ sạch của khâu trích xuất bounding box ban đầu từ detector.`

**4. Một chỗ bạn đúng và ReID sai (frame, ID, vì sao):**

`Frame: 107–113 (kéo dài xuyên suốt frame 16–116).

ID: ID 7 của mô hình ReID.

Vì sao: Mô hình ReID liên tục phát hiện và duy trì một bounding box (T7) bao quanh một ki-ốt/biển quảng cáo cố định đặt dưới chân cột biển báo giao thông ở dải phân cách. Đây là một cấu trúc tĩnh, không phải phương tiện giao thông di động; nhãn người gán đã loại trừ chính xác vật thể này.`

**5. Một chỗ ReID làm bạn xem lại annotation (frame, ID, vì sao), hoặc lý do evidence cho thấy model sai:**

`Frame: 87–95.

ID: Track gold 5 / ReID ID 18.

Vì sao: Báo cáo so khớp chỉ ra ReID đã bắt đầu nhận diện và gắn box cho xe sedan từ frame 87, trong khi nhãn tự gán ban đầu chỉ bắt đầu xuất hiện từ frame 96 (bị sót 8 frame đầu). Khi đối chiếu lại video gốc, đầu xe đã bắt đầu lộ diện từ làn đường phía xa tại frame 87. Đây là bằng chứng cho thấy mô hình phát hiện sớm hơn người gán, giúp bổ sung chính xác đoạn mở đầu của track ở bước rework.`

## 6. Nếu phải gán thêm 10 clip nữa

Bạn sẽ sửa gì trong `GUIDELINE_MINI.md`, và đổi gì trong quy trình làm việc của mình?

`Quy định rõ tiêu chí cắt rìa (outside): Chỉ tạo track khi phần thân xe lộ diện $\ge 20\%$ và phải bấm outside ở frame liền kề khi diện tích xe còn lại $< 10\%$ để giải quyết dứt điểm lỗi bbox treo.Quy chuẩn khoảng cách keyframe: Đặt keyframe cách nhau tối đa 12–15 frame đối với xe chạy thẳng đều, và giảm xuống 3–5 frame khi xe chuyển làn, giảm tốc hoặc đi qua vùng bị che khuất để hạn chế sụt giảm IoU do trôi bbox.Phổ biến danh mục ngoại trừ (False Positive blacklist): Liệt kê rõ các trạm chờ xe buýt, ki-ốt, pano quảng cáo ven đường không được gán vào các class phương tiện giao thông.Đổi trong quy trình làm việc:Ứng dụng luồng Model-Assisted Annotation: Sử dụng mô hình chạy trước để tạo khung bounding box sơ bộ, sau đó người gán tập trung vào việc chuẩn hóa ID, xóa bỏ các FP từ vật thể tĩnh và tinh chỉnh tiếp xúc viền.Thiết lập script sanity check cục bộ: Chạy mã kiểm tra tự động trước khi đóng gói dữ liệu để quét nhanh các lỗi ID switch bất thường, kiểm tra bbox đứng yên nhiều frame và phát hiện các track chưa được đánh dấu outside ở frame kết thúc.`

## 7. Tệp đã nộp

- [ ] `annotations/clip_01/gt.txt`
- [ ] `annotations/clip_02/gt.txt`
- [ ] `evidence/pre-gold/clip_01/gt.txt` và `manifest.json`
- [ ] `GUIDELINE_MINI.md` đã điền
- [ ] `outputs/eval_vs_gold.json`
- [ ] `outputs/model_bytetrack_clip_01.txt`
- [ ] `outputs/model_reid_clip_01.txt`
- [ ] `outputs/model_run_config.json`
- [ ] `outputs/eval_bytetrack_vs_gold.json`, `outputs/eval_reid_vs_gold.json`, `outputs/eval_reid_vs_me.json`
- [ ] `reports/review_partner.md`
- [ ] `reports/REPORT.md` (file này)
