import  numpy as np
import random
ouji = np.loadtxt('oujilide_bbb.txt')
print(ouji.shape)
# print(ouji[0])
_rand1 = np.random.normal(0, 1, (128,128))
_rand2 = np.random.normal(0, 1, (128,128))
def my_gramSchmidt_np(vectors):
    def proj(x, u):
        u = unit_vec(u)
        return np.dot(x, u) * u
    def unit_vec(x):
        return x / np.linalg.norm(x)
    vectors = np.atleast_2d(vectors)
    if len(vectors) == 0:
        return []
    if len(vectors) == 1:
        return unit_vec(vectors)

    u = vectors[-1]

    basis = my_gramSchmidt_np(vectors[0:-1])
    temp=[]
    for v in basis:
        temp.append(proj(u, v))
    w = np.atleast_2d(u - np.sum(temp,axis=0  ))
    basis = np.append(basis, unit_vec(w), axis=0)
    return basis

def gram_schmidt_np(V):
 # YOUR CODE HERE
    if type(V) is not np.ndarray:
        raise ValueError
    else:
        if V.shape[0] != V.shape[1]:
            raise ValueError

    vectors = np.array(V)
    vectors = np.transpose(vectors)
    vectors = vectors.tolist()
    vectors = my_gramSchmidt_np(vectors)
    dim = np.linalg.matrix_rank(vectors)


    if dim == vectors.shape[0]:
        return vectors
    else:
        vectors = np.transpose(vectors)
        for i in vectors:
            for pos, j in enumerate(i):
                if pos >= dim:
                    i[pos] = 0
        return vectors

smith_v1=gram_schmidt_np(_rand1)
smith_v2=gram_schmidt_np(_rand2)
ans=[]
for i in range(len(ouji)):
    vect=[]
    for j in range(len(smith_v1)):
        temp=np.dot(ouji[i],smith_v1[j])
        if temp>0:
            temp=1
        else:
            temp=0
        vect.append(temp)

    for j in range(len(smith_v2)):
        temp = np.dot(ouji[i], smith_v2[j])
        if temp > 0:
            temp = 1
        else:
            temp = 0
        vect.append(temp)
    vect=np.array(vect)
    ans.append(vect)

# print(len(ans))
# print(ans[0])
def int_count(n1,n2):
    if len(n1)!=len(n2):
        print("not matching when xor")
    num=0
    for i in range(len(n1)):
        if n1[i]!=n2[i]:
            num+=1
    return num

A=np.zeros(257)

def extract_elements(array, indices):
    new_array = [array[i] for i in indices]
    return new_array


# 定义一个空字典用于保存数据
data_label = {}
# 打开文件
with open('oujilide.txt', 'r') as file:
    for line_num, line in enumerate(file):
        # 奇数行处理文本
        if line_num % 3 == 0:
            text = line.strip()
            result = text.split('/')[0]
            # print("文本:", result)
            data_label[int(line_num/3)]=result
        # 偶数行处理数组
        elif line_num % 3 == 1:
            features = list(map(float, line.strip().split()))  # 将字符串数组转换为浮点数列表
            # 将标签和特征保存到字典中
             #data_dict[result] = features
        # elif line_num % 3 == 2:
            # text_2 = line.strip()
# print(len(data_label))
# print(data_label[0])
# 随机选取两个元素，组合成t位字符串

def find_adjacent_duplicates(labels):
    adjacent_indices = []

    # 遍历列表中的元素，从第二个元素开始比较
    for i in range(1, len(labels)):
        # 检查当前元素是否与前一个元素相同
        if labels[i] == labels[i - 1]:
            # 如果相同，则将当前下标和前一个下标添加到相邻下标列表中
            adjacent_indices.append(i)
            adjacent_indices.append(i - 1)

    return adjacent_indices

# 查找相邻下标所对应值相同的所有下标
result_indices = find_adjacent_duplicates(data_label)
print("相邻下标所对应值相同的所有下标:", result_indices)

def select_combination(lst1,lst2, t):
    num=range(len(lst1))
    num_sample=random.sample(num, t)
    element1 = extract_elements(lst1, num_sample)
    element2 = extract_elements(lst2, num_sample)
    return element1, element2

def select_N_combination(lst_list, t):
    num=range(len(lst_list[0]))
    num_sample=random.sample(num, t)
    element_list =[]
    for i in range(len(lst_list)):
        element_list.append( extract_elements(lst_list[i], num_sample))

    return element_list


# 验证是否有m次组合相同
def validate_combinations(lst1, lst2, t, N, m):
    count=0
    for _ in range(N):
        combination1, combination2 = select_combination(lst1, lst2,t)
        if combination1==combination2:
            count+=1
            # print(len(combination1))
            # print(len(combination2))
            # print("c1:",combination1)
            # print("c2:", combination2)
        if count == m :
            return True, count
    return False, count

# 验证是否有m次组合相同
def validate_N_combinations(lst_list, t, N, m):
    count=0
    for _ in range(N):
        combination_list= select_N_combination(lst_list,t)
        if all_elments_equal_list(combination_list):
            count+=1
            # print(len(combination1))
            # print(len(combination2))
            # print("c1:",combination1)
            # print("c2:", combination2)
        if count == m :
            return True, count
    return False, count


def all_elments_equal_list(lst):
    if not lst:
        return True
    return len(set(map(tuple,lst))) == 1

def all_elments_equal_int(lst):
    if not lst:
        return True
    return len(set(lst)) == 1

# lst=[1,2,1]
# print(all_elments_equal_int(lst))
# lst=[1,1,1]
# print(all_elments_equal_int(lst))
# lst=[[1,2],[1,3]]
# print(all_elments_equal_list(lst))
# lst=[[1,2],[1,2]]
# print(all_elments_equal_list(lst))
def compare_pairs(ans_list, feature_list,pair1_index,pair2_index,t,N,m):
    # 随机选择两对索引
    # pair1_index = random.randint(0, len(ans_list) - 1)
    # pair2_index = random.randint(0, len(ans_list) - 1)

    # 获取随机选择的两对
    pair1_ans = ans_list[pair1_index]
    pair1_feature = feature_list[pair1_index]
    pair2_ans = ans_list[pair2_index]
    pair2_feature = feature_list[pair2_index]

    # 输出两对的答案和特征
    # print("Pair 1:")
    # print("Answer:", pair1_ans)
    # print("Feature:", pair1_feature)
    # print()
    # print("Pair 2:")
    # print("Answer:", pair2_ans)
    # print("Feature:", pair2_feature)
    # t=10
    # N=128
    # m=2
    flag, count= validate_combinations(pair1_ans, pair2_ans, t, N, m)
    # print("flag：",flag)
    # print("count:",count)
    TP=0
    TN=0
    FP=0
    FN=0
    # 比较两对答案和特征
    if flag==True and (pair1_feature==pair2_feature):
        # print("The answers of pair 1 and pair 2 are the same. ",pair1_index)
        TP+=1
    elif  flag==False and (pair1_feature==pair2_feature):
        FP+=1
    elif flag==False and (pair1_feature!=pair2_feature):
        # print("The answers of pair 1 and pair 2 are different.",pair1_index)
        TN+=1
    else:
        FN+=1
    return TP,FP,FN,TN



def compare_Num_pairs(ans_list, feature_list,pair_index_list,t,N,m):
    # 随机选择两对索引
    # pair1_index = random.randint(0, len(ans_list) - 1)
    # pair2_index = random.randint(0, len(ans_list) - 1)

    # 获取随机选择的N对
    pair_ans_list=[]
    pair_feature_list = []
    for i in range(len(pair_index_list)):
        pair_ans_list.append(ans_list[pair_index_list[i]])
        pair_feature_list.append(feature_list[pair_index_list[i]])

    # 输出两对的答案和特征
    # print("Pair 1:")
    # print("Answer:", pair1_ans)
    # print("Feature:", pair1_feature)
    # print()
    # print("Pair 2:")
    # print("Answer:", pair2_ans)
    # print("Feature:", pair2_feature)
    # t=10
    # N=128
    # m=2
    flag, count= validate_N_combinations(pair_ans_list, t, N, m)
    # print("flag：",flag)
    # print("count:",count)
    TP=0
    TN=0
    FP=0
    FN=0
    # 比较两对答案和特征
    if flag==True and (all_elments_equal_int(pair_feature_list)):
        # print("The answers of pair 1 and pair 2 are the same. ",pair1_index)
        TP+=1
    elif  flag==False and (all_elments_equal_int(pair_feature_list)):
        FP+=1
    elif flag==False and (all_elments_equal_int(pair_feature_list)==False):
        # print("The answers of pair 1 and pair 2 are different.",pair1_index)
        TN+=1
    else:
        FN+=1
    return TP,FP,FN,TN

for t in [4,5,6]:
    for N in [64,96,128]:
        for m in [2]:
            countcc=len(ans)-4
            co=5
            total_ans = [[0 for _ in range(countcc+1)] for _ in range(co+1)]
            cou_TN_TP = 0
            for i in range(countcc):
                Total_TP = 0
                Total_FP = 0
                Total_FN = 0
                Total_TN = 0
                ccc = 0
                for tt in [co]:
                    TP,FP,FN,TN=compare_Num_pairs(ans,data_label,[i,i+1,i+2,i+3,i+4],t,N,m)
                    Total_TP += TP
                    Total_FP += FP
                    Total_FN += FN
                    Total_TN += TN
                    if TN==1 or TP==1:
                        ccc = 1
                if ccc==1:
                    cou_TN_TP+=1
            print("t:",t)
            print("N:", N)
            print("M:", m)
            print("Total_ture:", (cou_TN_TP) / len(ans))
                    # print(TP*1+FP*2+FN*4+TN*8)
                # print("Parameters:", t,N,m)
                # print("TP:",Total_TP)
                # print("TN:",Total_TN)
                # print("Total_ture:",(Total_TN+Total_TP)/len(ans))
            # print(total_ans[2][6])
            # print(total_ans[1][1])
            #
            # for i in range(countcc):
            #     ccc=0
            #     for j in range(co):
            #         # print(total_ans[j][i])
            #         if total_ans[j][i] == 8 or total_ans[j][i] == 1:
            #             ccc=1
            #     if ccc==1 :
            #         cou_TN_TP+=1


            # # 打开文件以追加写入变量
            # with open("output_file5.txt", 'a') as file:
            #     # 将变量追加写入文件
            #     file.write(f"t: {t}\n")
            #     file.write(f"N: {N}\n")
            #     file.write(f"m: {m}\n")
            #     # 将变量追加写入文件
            #     file.write(f"TP: {TP}\n")
            #     file.write(f"TN: {TN}\n")
            #     file.write(f"Total_true: {(Total_TN+Total_TP)/len(ans)}\n")



# for i in range(len(ans)):
#     print(ans[i])
# for i in range(len(ans)):
#
#     n=int_count(ans[7220],ans[i])
#     A[n]+=1
# print(A)
# np.savetxt("len_pro.txt",A)
# for i in range(60):
#     print(int_count(ans[7220],ans[7220+i]))


# # 原始数组
# original_array = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
#
# # 想要提取的索引
# indices_to_extract = [4,3,2,1]

# 提取指定索引处的元素
