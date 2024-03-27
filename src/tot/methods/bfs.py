import itertools
import numpy as np
import tree_of_evaluator as toe
from functools import partial
from src.tot.models import gpt

def get_value(task, x, y, n_evaluate_sample, cache_value=True): # 根據任務、輸入 x、輸出 y 和評估樣本數量來獲得單層單一節點的值
    value_prompt = task.value_prompt_wrap(x, y) # 生成值提示
    if cache_value and value_prompt in task.value_cache:  # 如果啟用值緩存且值提示已經存在於任務的值緩存中，則從緩存中返回值
        return task.value_cache[value_prompt]
    value_outputs = gpt(value_prompt, n=n_evaluate_sample, stop=None) # 通過 GPT 模型獲取值輸出列表
    value = task.value_outputs_unwrap(x, y, value_outputs) # 解析值輸出列表以獲取最終值
    if cache_value: # 如果啟用值緩存，則將值緩存到任務中
        task.value_cache[value_prompt] = value
    print(f'\nvalue: {value}\n')
    return value

def get_values(task, x, ys, n_evaluate_sample, args, cache_value=True): # 得到該層所有節點的值，ys 為 solve 裡的 new_ys
    values = [] # 儲存每個部分輸出的值
    local_value_cache = {} # 用於本地緩存的字典
    eva_layer_nodes_infos = [] # 存放一層內所有節點 tree 的地方
    if args.method_evaluate == 'value':  
        for y in ys:  # each partial output
            if y in local_value_cache:  # avoid duplicate candidates
                value = 0
            else:    
                value = get_value(task, x, y, n_evaluate_sample, cache_value=cache_value) # 獲取值並加入到本地緩存中
                local_value_cache[y] = value
            values.append(value) # 將值加入到結果列表中
    elif args.method_evaluate == 'tree':
        for y in ys:  # each partial output
            if y in local_value_cache:  # avoid duplicate candidates
                value = 0
            else:    
                value, eva_one_node_infos = toe.Evaluator_Tree(task, x, y, n_evaluate_sample, cache_value=cache_value) # 獲取值和單一節點 tree 的資訊
                print(f'value: {value}\neva_one_node_infos: {eva_one_node_infos}\n')
                local_value_cache[y] = value
            values.append(value) # value 存入 list
            eva_layer_nodes_infos.append(eva_one_node_infos) # 單一節點 tree 的資訊存入一層內所有節點 tree 的 list
    return values, eva_layer_nodes_infos # 回傳該層所有 value 和 tree 的資訊

def get_votes(task, x, ys, n_evaluate_sample): # 根據輸入和多個部分輸出計算投票
    vote_prompt = task.vote_prompt_wrap(x, ys) # 獲取投票提示
    vote_outputs = gpt(vote_prompt, n=n_evaluate_sample, stop=None) # 使用GPT生成投票結果
    values = task.vote_outputs_unwrap(vote_outputs, len(ys)) # 解析投票結果
    return values # 返回投票結果列表

def get_proposals(task, x, y): # 獲取提議
    propose_prompt = task.propose_prompt_wrap(x, y) # 獲取提議提示
    proposals = gpt(propose_prompt, n=1, stop=None)[0].split('\n') # 使用GPT生成提議
    print(f'Propose response: {proposals}\n')
    return [y + _ + '\n' for _ in proposals] # 返回提議列表

def get_samples(task, x, y, n_generate_sample, prompt_sample, stop): # 根據提供的提示樣本類型選擇不同的提示方式
    if prompt_sample == 'standard':
        prompt = task.standard_prompt_wrap(x, y) # 使用標準提示格式生成提示文本
    elif prompt_sample == 'cot':
        prompt = task.cot_prompt_wrap(x, y) # 使用 cot 提示格式生成提示文本
    else:
        raise ValueError(f'prompt_sample {prompt_sample} not recognized') # 如果提示樣本類型無法識別，則拋出ValueError
    samples = gpt(prompt, n=n_generate_sample, stop=stop) # 使用GPT模型生成指定數量的樣本文本
    return [y + _ for _ in samples] # 返回生成的樣本列表，並將部分輸出文本 y 添加到每個樣本文本中，此為 generator的最終產物

def solve(args, task, idx, to_print=True): # 解決24點遊戲任務的函式
    global gpt
    gpt = partial(gpt, model=args.backend, temperature=args.temperature)
    print(f'gpt: {gpt}\n')
    x = task.get_input(idx)  # input，最初輸入的四個數字
    ys = ['']  # current output candidates，本層輸出值
    infos = [] # 信息列表，用於記錄每一步的信息
    new_ys = [''] # 所有舊層+下層輸出值
    eva_all_nodes_infos = []
    for step in range(task.steps):
        # generation
        if args.method_generate == 'sample':
            new_ys = [get_samples(task, x, y, args.n_generate_sample, prompt_sample=args.prompt_sample, stop=task.stops[step]) for y in ys] # 使用 get_samples 函式生成新的候選輸出列表
            print(f'ys: {ys}\nnew_ys: {new_ys}\n')
        elif args.method_generate == 'propose':
            new_ys = [get_proposals(task, x, y) for y in ys] # 使用 get_proposals 函式生成新的候選輸出列表
        new_ys = list(itertools.chain(*new_ys)) # 將生成的所有候選輸出列表合併成一個列表
        ids = list(range(len(new_ys))) # 為每個候選輸出賦予唯一的 ID
        print('\nGenerator Finish!\n')
        # evaluation，values為 new_ys 每一項的值
        if args.method_evaluate == 'vote':
            values = get_votes(task, x, new_ys, args.n_evaluate_sample) # 使用 get_votes 函式評估每個候選輸出的價值
        elif args.method_evaluate == 'value' or args.method_evaluate == 'tree':
            values, eva_layer_nodes_infos = get_values(task, x, new_ys, args.n_evaluate_sample, args.method_evaluate) # 使用 get_values 函式得出該層所有節點的值和 tree 資訊
            print(f'values: {values}\neva_layer_nodes_infos: {eva_layer_nodes_infos}\n')
            eva_all_nodes_infos.append(eva_layer_nodes_infos) # 該層 tree 的資訊加入全部 tree 資訊的 list
        print('\nEvaluator Finish!\n')
        # selection
        if args.method_select == 'sample': # 根據價值比例進行抽樣，選擇一部分候選輸出作為下一步的輸出
            ps = np.array(values) / sum(values) 
            select_ids = np.random.choice(ids, size=args.n_select_sample, p=ps).tolist()
        elif args.method_select == 'greedy':
            select_ids = sorted(ids, key=lambda x: values[x], reverse=True)[:args.n_select_sample] # 根據價值從大到小排序，選擇價值最高的一部分候選輸出作為下一步的輸出
        select_new_ys = [new_ys[select_id] for select_id in select_ids] # 從所有候選輸出中選擇出被選中的輸出

        # log
        if to_print: 
            sorted_new_ys, sorted_values = zip(*sorted(zip(new_ys, values), key=lambda x: x[1], reverse=True)) # 打印生成的候選輸出、其對應的價值以及被選中的輸出
            print(f'-- new_ys --: {sorted_new_ys}\n-- sol values --: {sorted_values}\n-- choices --: {select_new_ys}\n')
        
        infos.append({'step': step, 'x': x, 'ys': ys, 'new_ys': new_ys, 'values': values, 'select_new_ys': select_new_ys}) # 將當前步驟的信息加入信息列表
        ys = select_new_ys # 更新當前輸出列表
    
    if to_print: # 打印最終解答
        print(f'all ys: {ys}\n')
    return ys, {'steps': infos},  {'eva_steps': eva_all_nodes_infos}# 返回最終解答和相關的信息字典

def naive_solve(args, task, idx, to_print=True):
    global gpt
    gpt = partial(gpt, model=args.backend, temperature=args.temperature)
    print(gpt)
    x = task.get_input(idx)  # input，獲取索引為 idx 的輸入
    ys = get_samples(task, x, '', args.n_generate_sample, args.prompt_sample, stop=None) # 獲取候選輸出列表 ys，使用的是 get_samples 函式
    return ys, {} # 返回候選輸出列表 ys，以及空字典作為額外信息