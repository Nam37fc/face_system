import numpy as np
from scipy.spatial import distance as dist
from collections import OrderedDict

class CentroidTracker:
    """
    Bộ theo dõi khuôn mặt đơn giản dựa trên khoảng cách Euclide giữa các trung tâm (Centroids).
    Gán ID duy nhất và bền vững cho khuôn mặt qua các frame.
    """
    def __init__(self, max_disappeared=10):
        self.next_id = 0
        self.objects = OrderedDict() # {object_id: centroid}
        self.disappeared = OrderedDict() # {object_id: frame_count}
        self.max_disappeared = max_disappeared

    def register(self, centroid):
        self.objects[self.next_id] = centroid
        self.disappeared[self.next_id] = 0
        self.next_id += 1
        return self.next_id - 1

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, rects):
        """
        Cập nhật vị trí và trả về danh sách IDs cho các hình chữ nhật hiện tại.
        rects: Danh sách [x1, y1, x2, y2]
        """
        if len(rects) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return {}

        # Tính toán centroids cho rects hiện tại
        input_centroids = np.zeros((len(rects), 2), dtype="int")
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            input_centroids[i] = (cX, cY)

        if len(self.objects) == 0:
            res_ids = []
            for i in range(0, len(input_centroids)):
                res_ids.append(self.register(input_centroids[i]))
            return {i: res_ids[i] for i in range(len(res_ids))}

        # Khớp centroids cũ với centroids mới
        object_ids = list(self.objects.keys())
        object_centroids = list(self.objects.values())

        D = dist.cdist(np.array(object_centroids), input_centroids)
        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        used_rows = set()
        used_cols = set()
        
        mapping = {} # {rect_index: object_id}

        for (row, col) in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue

            object_id = object_ids[row]
            self.objects[object_id] = input_centroids[col]
            self.disappeared[object_id] = 0
            mapping[col] = object_id
            
            used_rows.add(row)
            used_cols.add(col)

        unused_rows = set(range(0, D.shape[0])).difference(used_rows)
        unused_cols = set(range(0, D.shape[1])).difference(used_cols)

        # Xử lý các đối tượng biến mất
        for row in unused_rows:
            object_id = object_ids[row]
            self.disappeared[object_id] += 1
            if self.disappeared[object_id] > self.max_disappeared:
                self.deregister(object_id)

        # Đăng ký đối tượng mới
        for col in unused_cols:
            mapping[col] = self.register(input_centroids[col])

        return mapping
