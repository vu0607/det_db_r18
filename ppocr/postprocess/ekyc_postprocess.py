import Levenshtein

IDENTITY_FRONT = [
    "CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM",
    "Độc lập - Tự do - Hạnh phúc",
    "GIẤY CHỨNG MINH NHÂN DÂN",
    "SỐ",
    "Họ tên:",
    "Sinh ngày",
    "Nguyên quán:",
    "Nơi ĐKHK thường trú:",
]

IDENTITY_BACK = [
    "Dân tộc:",
    "Tôn giáo:",
    "NGÓN TRỎ TRÁI",
    "NGÓN TRỎ PHẢI",
    "DẤU VẾT RIÊNG VÀ DỊ HÌNH",
    "GIÁM ĐỐC CA",
]

CITIZEN_V1_FRONT = [
    "CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM",
    "Độc lập - Tự do - Hạnh phúc",
    "CĂN CƯỚC CÔNG DÂN",
    "Số:",
    "Họ và tên:",
    "Ngày, tháng, năm sinh:",
    "Giới tính:",
    "Quốc tịch:",
    "Quê quán:",
    "Nơi thường trú:",
    "Có giá trị đến:"
]

CITIZEN_V1_BACK = [
    "Đặc điểm nhận dạng:",
    "NGÓN TRỎ TRÁI",
    "NGÓN TRỎ PHẢI",
    "CỤC TRƯỞNG CỤC CẢNH SÁT",
    "QUẢN LÝ HÀNH CHÍNH VỀ TRẬT TỰ XÃ HỘI",
    "ĐKQL CƯ TRÚ VÀ DLQG VỀ DÂN CƯ",
]

CITIZEN_V2_FRONT = [
    "CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM",
    "Độc lập - Tự do - Hạnh phúc",
    "SOCIALIST REPUBLIC OF VIET NAM",
    "Independence - Freedom - Happiness",
    "CĂN CƯỚC CÔNG DÂN",
    "Citizen Identity Card",
    "Số / No.:",
    "Họ và tên / Full name:"
    "Ngày sinh / Date of birth:",
    "Giới tính / Sex:",
    "Quốc tịch / Nationality:",
    "Quê quán / Place of origin:",
    "Nơi thường trú / Place of residence:",
    "Có giá trị đến:",
    "Date of expiry",
]

CITIZEN_V2_BACK = [
    "Đặc điểm nhận dạng / Persional identification:",
    "Ngày, tháng, năm / Date, month, year:",
    "Ngón trỏ trái",
    "Ngón trỏ phải",
    "Left index finger",
    "Right index finger",
    "CỤC TRƯỞNG CỤC CẢNH SÁT",
    "QUẢN LÝ HÀNH CHÍNH VỀ TRẬT TỰ XÃ HỘI",
]

ID_CARD_KEY_WORDS = IDENTITY_FRONT + IDENTITY_BACK + \
                    CITIZEN_V1_FRONT + CITIZEN_V1_BACK + \
                    CITIZEN_V2_FRONT + CITIZEN_V2_BACK


class EKYCPostProcess(object):
    def __init__(self, cer_threshold=0.15):
        self.cer_threshold = cer_threshold

    def __call__(self, pred):
        post_processed_pred = pred
        min_edit_dis = 1
        for target in ID_CARD_KEY_WORDS:
            edit_dis = Levenshtein.distance(pred, target) / max(
                len(pred), len(target), 1)

            if edit_dis < self.cer_threshold and edit_dis < min_edit_dis:
                post_processed_pred = target

        return post_processed_pred

