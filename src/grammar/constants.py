"""
FAST-PLATE-OCR-VIETNAMESE: Constants & Reference Data
Quy chuẩn biển số xe Việt Nam theo TT 24/2023/TT-BCA & TT 169/2021/TT-BQP
"""

# Tập ký tự số
DIGITS = "0123456789"

# 20 chữ cái dân sự tiêu chuẩn (Điều 37 Thông tư 24/2023/TT-BCA)
CIVILIAN_LETTERS = "ABCDEFGHKLMNPSTUVXYZ"

# 22 chữ cái Latinh hợp lệ toàn bộ hệ thống (dân sự + 'Q' quân sự/QT + 'R' rơ-moóc)
VALID_LETTERS = "ABCDEFGHKLMNPQRSTUVXYZ"

# Bảng chữ cái phân lớp đầy đủ cho mô hình OCR (32 ký tự alphanumeric + padding '_')
ALPHABET = "0123456789ABCDEFGHKLMNPQRSTUVXYZ_"
PAD_CHAR = "_"

# Các ký tự bị loại trừ tuyệt đối
EXCLUDED_LETTERS = {"I", "O", "J", "W"}

# Danh mục 63 tỉnh/thành phố và Cục CSGT (Tổng cộng các mã hợp lệ từ 11 - 99)
VALID_PROVINCE_CODES = {
    "11": "Cao Bằng",
    "12": "Lạng Sơn",
    "14": "Quảng Ninh",
    "15": "Hải Phòng",
    "16": "Hải Phòng",
    "17": "Thái Bình",
    "18": "Nam Định",
    "19": "Phú Thọ",
    "20": "Thái Nguyên",
    "21": "Yên Bái",
    "22": "Tuyên Quang",
    "23": "Hà Giang",
    "24": "Lào Cai",
    "25": "Lai Châu",
    "26": "Sơn La",
    "27": "Điện Biên",
    "28": "Hòa Bình",
    "29": "Hà Nội",
    "30": "Hà Nội",
    "31": "Hà Nội",
    "32": "Hà Nội",
    "33": "Hà Nội",
    "34": "Hải Dương",
    "35": "Ninh Bình",
    "36": "Thanh Hóa",
    "37": "Nghệ An",
    "38": "Hà Tĩnh",
    "39": "Đồng Nai",
    "40": "Hà Nội",
    "43": "Đà Nẵng",
    "47": "Đắk Lắk",
    "48": "Đắk Nông",
    "49": "Lâm Đồng",
    "50": "TP. Hồ Chí Minh",
    "51": "TP. Hồ Chí Minh",
    "52": "TP. Hồ Chí Minh",
    "53": "TP. Hồ Chí Minh",
    "54": "TP. Hồ Chí Minh",
    "55": "TP. Hồ Chí Minh",
    "56": "TP. Hồ Chí Minh",
    "57": "TP. Hồ Chí Minh",
    "58": "TP. Hồ Chí Minh",
    "59": "TP. Hồ Chí Minh",
    "60": "Đồng Nai",
    "61": "Bình Dương",
    "62": "Long An",
    "63": "Tiền Giang",
    "64": "Vĩnh Long",
    "65": "Cần Thơ",
    "66": "Đồng Tháp",
    "67": "An Giang",
    "68": "Kiên Giang",
    "69": "Cà Mau",
    "70": "Tây Ninh",
    "71": "Bến Tre",
    "72": "Bà Rịa - Vũng Tàu",
    "73": "Quảng Bình",
    "74": "Quảng Trị",
    "75": "Thừa Thiên Huế",
    "76": "Quảng Ngãi",
    "77": "Bình Định",
    "78": "Phú Yên",
    "79": "Khánh Hòa",
    "80": "Cục Cảnh sát giao thông (Cấp Trung ương)",
    "81": "Gia Lai",
    "82": "Kon Tum",
    "83": "Sóc Trăng",
    "84": "Trà Vinh",
    "85": "Ninh Thuận",
    "86": "Bình Thuận",
    "88": "Vĩnh Phúc",
    "89": "Hưng Yên",
    "90": "Hà Nam",
    "92": "Quảng Nam",
    "93": "Bình Phước",
    "94": "Bạc Liêu",
    "95": "Hậu Giang",
    "97": "Bắc Kạn",
    "98": "Bắc Giang",
    "99": "Bắc Ninh",
}

# Danh mục ký hiệu xe quân sự (Phụ lục III Thông tư 169/2021/TT-BQP)
MILITARY_UNIT_CODES = {
    # Cơ quan Bộ Quốc phòng & Tổng cục
    "TM": "Bộ Tổng Tham mưu - Cơ quan BQP",
    "TC": "Tổng cục Chính trị",
    "TH": "Tổng cục Hậu cần",
    "TT": "Tổng cục Kỹ thuật",
    "TK": "Tổng cục Công nghiệp quốc phòng",
    "TN": "Tổng cục II (Tổng cục Tình báo)",
    # Quân khu, Quân chủng, Bộ Tư lệnh
    "KA": "Quân khu 1",
    "KB": "Quân khu 2",
    "KC": "Quân khu 3",
    "KD": "Quân khu 4",
    "KV": "Quân khu 5",
    "KP": "Quân khu 7",
    "KK": "Quân khu 9",
    "KT": "Bộ Tư lệnh Thủ đô Hà Nội",
    "QA": "Quân chủng Phòng không - Không quân",
    "QH": "Quân chủng Hải quân",
    "QB": "Bộ Tư lệnh Bộ đội Biên phòng",
    "QC": "Bộ Tư lệnh Cảnh sát biển",
    "QM": "Bộ Tư lệnh Tác chiến không gian mạng (BTL 86)",
    "BL": "Bộ Tư lệnh Bảo vệ Lăng Chủ tịch Hồ Chí Minh",
    # Binh chủng
    "BB": "Binh chủng Tăng - Thiết giáp",
    "BC": "Binh chủng Công binh",
    "BK": "Binh chủng Đặc công",
    "BP": "Binh chủng Pháo binh",
    "BH": "Binh chủng Hóa học",
    "BT": "Binh chủng Thông tin liên lạc",
    # Quân đoàn, Học viện
    "AA": "Quân đoàn 12 (trước là Quân đoàn 1)",
    "AB": "Quân đoàn 2",
    "AC": "Quân đoàn 3",
    "AD": "Quân đoàn 4",
    "HA": "Học viện Quốc phòng",
    "HB": "Học viện Lục quân",
    "HC": "Học viện Chính trị",
    "HE": "Học viện Hậu cần",
    "HD": "Học viện Kỹ thuật quân sự",
    "HH": "Học viện Quân y",
    "HT": "Trường Sĩ quan Lục quân 1",
    # Viện, Bệnh viện, Binh đoàn & Doanh nghiệp quân đội
    "PQ": "Viện Khoa học và Công nghệ quân sự",
    "PM": "Viện Thiết kế Bộ Quốc phòng",
    "PX": "Trung tâm Nhiệt đới Việt - Nga",
    "PP": "Bệnh viện Quân y trực thuộc BQP",
    "AV": "Binh đoàn 11",
    "AT": "Binh đoàn 12",
    "AN": "Binh đoàn 15",
    "AX": "Binh đoàn 16",
    "AM": "Binh đoàn 18",
    "VT": "Tập đoàn CN - Viễn thông Quân đội (Viettel)",
    "CA": "Tổng công ty 36",
    "CB": "Ngân hàng TMCP Quân đội (MB)",
    "CD": "Tổng công ty Vạn Xuân",
    "CH": "Tổng công ty Đông Bắc",
    "CM": "Tổng công ty Thái Sơn",
    "CN": "Tổng công ty Đầu tư phát triển Nhà và Đô thị BQP",
    "CP": "Tổng công ty 319",
    "CT": "Công ty TNHH MTV Ứng dụng Kỹ thuật và Sản xuất (Tecapro)",
    "CV": "Tổng công ty Xây dựng Lũng Lô",
}

# Danh mục sê-ri ngoại giao và tổ chức quốc tế
DIPLOMATIC_SERIES = {
    "NG": "Đại sứ quán, Cơ quan lãnh sự, Viên chức ngoại giao",
    "QT": "Cơ quan đại diện Tổ chức Quốc tế và Viên chức quốc tế",
    "NN": "Tổ chức, văn phòng đại diện, cá nhân người nước ngoài",
    "CV": "Nhân viên hành chính kỹ thuật của cơ quan ngoại giao",
}

# Danh mục sê-ri chuyên dùng đặc thù
SPECIAL_SERIES = {
    "LD": "Xe liên doanh",
    "DA": "Xe dự án nước ngoài",
    "R":  "Rơ-moóc / Sơ-mi rơ-moóc",
    "RM": "Rơ-moóc / Sơ-mi rơ-moóc (mở rộng)",
    "MK": "Máy kéo, máy chuyên dùng",
    "MD": "Xe máy điện",
    "CD": "Xe 4 bánh gắn động cơ",
    "HC": "Xe hoạt động hạn chế",
    "T":  "Xe đăng ký tạm thời",
}
