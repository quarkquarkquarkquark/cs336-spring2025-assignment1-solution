from collections import Counter

cpdef find_max_pairs_cython(dict pairs_cnt, dict vocab):
    # 使用C类型来存储这些值，避免Python对象开销
    cdef long long max_cnt = -1
    cdef tuple max_pair = (-1, -1)
    cdef tuple max_pair_str = None
    cdef long long count
    cdef tuple pair, current_pair_str

    # .items() 调用是Python操作，但循环内部可以被优化
    for pair, count in pairs_cnt.items():
        if count > max_cnt:
            max_cnt = count
            max_pair = pair
            max_pair_str = (vocab[pair[0]], vocab[pair[1]])
        elif count == max_cnt:
            current_pair_str = (vocab[pair[0]], vocab[pair[1]])
            if current_pair_str > max_pair_str:
                max_pair = pair
                max_pair_str = current_pair_str

    return max_pair, max_cnt

cpdef replace_pair(list lst,tuple pair,long long new_val):

    cdef:
        list res
        long long a
        long long b
        long long i=0
        long long n=len(lst)

    a, b = pair
    res=[]
    i=0
    while i<n:
        if i!=n-1 and lst[i]==pair[0] and lst[i+1]==pair[1]:
            res.append(new_val)
            i+=2
            continue

        res.append(lst[i])
        i+=1
    return res

cpdef update_pairs_cnt(dict pairs_cnt, dict vocab,dict word_cnt,dict word_b_pairs,long long cnt,tuple merge_pair):
    cdef long long n=0
    cdef tuple p
    cdef list word_tokens
    cdef long long freq

    for k in word_cnt.keys():
        if vocab[cnt-1] in k:
            n=len(word_b_pairs[k])
            word_tokens=word_b_pairs[k]
            freq = word_cnt[k]

            for i in range(n-1):
                p=(word_tokens[i],word_tokens[i+1])
                pairs_cnt[p]-=freq
                if pairs_cnt[p]==0:
                    del pairs_cnt[p]

            word_tokens=replace_pair(word_b_pairs[k],merge_pair,cnt-1)
            n=len(word_tokens)
            word_b_pairs[k]=word_tokens
            for i in range(n-1):
                p=(word_tokens[i],word_tokens[i+1])
                if p in pairs_cnt:
                    pairs_cnt[p]+=freq
                else:
                    pairs_cnt[p]=freq

    return word_b_pairs,pairs_cnt

cpdef encode_apply_merges(dict merges_dict,list res):
    cdef long long a,b,rep_idx,min_a,min_b,min_rep_idx
    cdef n,f=1
    cdef list res_new=res
    while f==1:
        f=0
        n=len(res)
        min_rep_idx=int(1e8)
        for i in range(n-1):
            a=res[i]
            b=res[i+1]
            if (a,b) in merges_dict:
                rep_idx=merges_dict[(a,b)]
                if rep_idx<min_rep_idx:
                    min_rep_idx=rep_idx
                    min_a=a
                    min_b=b
                    f=1

        if f==1:
            res=replace_pair(res,(min_a,min_b),min_rep_idx)
    return res