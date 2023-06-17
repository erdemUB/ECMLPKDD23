from collections import deque

class Bucket:
    def __init__(self, mn_value, tot_bucket):
        self.cur_min_buck_id = mn_value
        self.total_bucket = tot_bucket
        self.bucket_array = {}
        self.bucket_map = {}
        self.processed_id = {}

    def initializeBucket(self, num_of_bucket):
        for cd in range(0, num_of_bucket + 1):
            self.bucket_array[cd] = deque([])
            self.bucket_map[cd] = {}

    def insertBucket(self, node_id, buck_id):
        self.bucket_map[buck_id][node_id] = len(self.bucket_array[buck_id])
        self.bucket_array[buck_id].append(node_id)

    def removeBucket(self, node_id, buck_id):
        node_loc_in_bucket = self.bucket_map[buck_id][node_id]
        last_node_in_array = self.bucket_array[buck_id][-1]
        self.bucket_map[buck_id][last_node_in_array] = node_loc_in_bucket
        self.bucket_array[buck_id][node_loc_in_bucket] = last_node_in_array

        self.bucket_map[buck_id].pop(node_id)
        self.bucket_array[buck_id].pop()

    def decCDValue(self, node_id, buck_id):
        # print("IN pop-->>>   ", node_id, buck_id, self.cur_min_buck_id)
        self.removeBucket(node_id, buck_id)
        self.insertBucket(node_id, buck_id - 1)
        self.cur_min_buck_id = min(buck_id - 1, self.cur_min_buck_id)

    def popMinFromBucket(self):
        while self.cur_min_buck_id < self.total_bucket and len(self.bucket_array[self.cur_min_buck_id]) == 0:
            self.cur_min_buck_id += 1

        if len(self.bucket_array[self.cur_min_buck_id]) == 0:
            return None, None
        cur_min = self.bucket_array[self.cur_min_buck_id].popleft()
        self.bucket_map[self.cur_min_buck_id].pop(cur_min)

        return cur_min, self.cur_min_buck_id


    def popMinBucket(self):
        # print("INSIDE POP MIN-->>>    ", self.cur_min_buck_id)
        while self.cur_min_buck_id <= self.total_bucket and len(self.bucket_array[self.cur_min_buck_id]) == 0:
            self.cur_min_buck_id += 1
            # print("OKK")
        # self.bucket_array[self.cur_min_buck_id].popleft()
        cur_min, cur_min_cd = None, None
        while self.cur_min_buck_id <= self.total_bucket and len(self.bucket_array[self.cur_min_buck_id]) > 0:
            cur_min = self.bucket_array[self.cur_min_buck_id].popleft()
            # print("INSIDE POP MIN-->>>    ",  self.cur_min_buck_id, cur_min)
            if cur_min not in self.processed_id:
                # self.processed_id[cur_min] = 1
                cur_min_cd = self.cur_min_buck_id
                if  len(self.bucket_array[self.cur_min_buck_id]) == 0 and self.cur_min_buck_id < self.total_bucket:
                    self.cur_min_buck_id += 1
                break
            if len(self.bucket_array[self.cur_min_buck_id]) == 0 and self.cur_min_buck_id < self.total_bucket:
                self.cur_min_buck_id += 1

        return cur_min, cur_min_cd

    def decVal(self, node_id, buck_id):
        # self.bucket_array[buck_id].popleft()
        self.bucket_array[buck_id - 1].append(node_id)
        self.cur_min_buck_id = min(buck_id - 1, self.cur_min_buck_id)
        # print(self.cur_min_buck_id)

if __name__ == '__main__':
    # array = [.42, .32, .33, .52, .37, .47, .51]
    id_array = [7, 1, 22, 10, 3, 5]
    val_array = [2, 1, 2, 2, 1, 1]
    cd = {518: 6, 209: 7, 648: 6, 622: 5, 719: 5, 506: 5, 714: 5, 575: 5, 540: 6, 420: 6, 716: 6, 585: 5, 528: 6, 584: 5}

    bs = Bucket(min(cd.values()), max(cd.values()))
    bs.initializeBucket(max(cd.values()))
    for v in cd.keys():
        bs.insertBucket(v, cd[v])

    # print(bs.bucket)
    while bs.cur_min_buck_id <= bs.total_bucket and bs.bucket_array[bs.cur_min_buck_id]:
        v, cd_v = bs.popMinBucket()
        print(v, cd_v)

