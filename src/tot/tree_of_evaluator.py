import itertools
import numpy as np
from src.tot.models import gpt

def Evaluator_Tree(task, x, y, args): # 輸入 task 類型(game 24)、x(題目的4個數字)、y(原本 tree 中單一節點的內容)、args(run 裡面的參數)
    print('\n-----Into Evaluator Tree-----\n')
    eva_infos = [] # evaluator tree 的所有資訊，每生完一棵會重置
    eva_new_ys = [''] # evaluator tree 新產生的節點存放的地方
    eva_ys = [''] # evaluator tree 舊節點存放的地方
    eva_select_new_ys = [''] # 存放第 2 層選到的節點
    eva_select_new_value = [] # 存放第 2 層選到的節點之 value
    node_values = [] # 存放選完後，3 個解加總後的 value
    for eva_step in range(0, 2): # 第 0 層為空，第 1 層為sure, likely, impossile，第 2 層為 sure, likely, impossile 的原因，各 3 個共 9 個節點
        # generate
        if eva_step == 0: #第 1 層先產出 sure, likely, impossile
            eva_new_ys = ['sure\n', 'likely\n', 'impossible\n']
        elif eva_step == 1: #第 2 層產出評價原因
            eva_new_ys = [get_prompt(task, x, y, eva_y) for eva_y in eva_ys] # 用 eva_ys 的每一項('sure\n', 'likely\n', 'impossible\n')拿去再生 3 個評價原因
            eva_new_ys = list(itertools.chain(*eva_new_ys)) # 將生成的所有候選輸出列表合併成一個列表
            ids = list(range(len(eva_new_ys))) # 為每個候選輸出賦予唯一的 ID
        print(f'eva_ys: {eva_ys}\neva_new_ys: {eva_new_ys}\n')
        print('\nGenerator of Evaluator Finish!\n')
        # evaluate
        eva_all_ys = [eva_y + eva_new_y for eva_y, eva_new_y in zip(eva_ys, eva_new_ys)] # 合併原本和新生層的文字
        eva_values = evaluator_get_values(task, x, y, eva_all_ys, args.n_evaluate_sample) # 得到該層所有節點的值，並加入 eva_values，而非覆蓋
        print(f'eva_values: {eva_values}\n')
        print('\nEvaluator of Evaluator Finish!\n')
        # choose
        if eva_step == 0:
            eva_infos.append({'eva_step': eva_step, 'x': x, 'y': y, 'eva_ys': eva_ys, 'eva_new_ys': eva_new_ys, 'eva_values': eva_values, 'eva_select_new_ys': eva_new_ys, 'eva_select_new_ys_values': eva_values, 'usage_so_far': gpt_usage(args.backend)}) # 寫入資訊
            eva_ys = ['sure\n', 'likely\n', 'impossible\n'] # 刷新 eva_ys
        elif eva_step == 1:
            eva_ps = np.array(eva_values) / sum(eva_values) # eva_values 轉換成 numpy 形式，eva_ps 為 eva_values 每一項的機率
            eva_select_ids = np.random.choice(ids, size = 3, p = eva_ps).tolist() # 從 9 個節點中根據每個節點不同的機率選出 3 個
            eva_select_new_ys.append(eva_new_ys[eva_select_id] for eva_select_id in eva_select_ids) # 選到的內容加入 list
            eva_select_new_value.append(eva_values[eva_select_id + 3] for eva_select_id in eva_select_ids) # 選到的 value 加入 list
            print(f'-- eva_new_ys --: {eva_new_ys}\n-- eva_sol_values --: {eva_values}\n-- eva_choices --: {eva_select_new_ys}\n-- eva_choices_values--: {eva_select_new_value}')
            # 寫入資訊，此為最終資訊
            eva_infos.append({'eva_step': eva_step, 'x': x, 'y': y, 'eva_ys': eva_ys, 'eva_new_ys': eva_new_ys, 'eva_values': eva_values, 'eva_select_new_ys': eva_select_new_ys, 'eva_select_new_ys_values': eva_select_new_value, 'usage_so_far': gpt_usage(args.backend)})
    print('\nChoice of Evaluator Finish!\n')
    # calculate node value
    for eva_select_id in eva_select_ids: # eva_values內容：id 0 ~ 2 為'sure\n', 'likely\n', 'impossible\n'的值，id 3 ~ 5 為 sure 生出 3 個節點的值，id 6 ~ 8 為 likely 生出 3 個節點的值，id 9 ~ 11 為 impossible 生出 3 個節點的值
        if eva_select_id >= 0 and eva_select_id <= 2: # eva_select_id 如果為 0 ~ 2，對應到 sure 生出的 3 個節點
            node_values.append(eva_values[0] + eva_values[eva_select_id + 3])
        if eva_select_id >= 3 and eva_select_id <= 5: # eva_select_id 如果為 3 ~ 5，對應到 likely 生出的 3 個節點
            node_values.append(eva_values[1] + eva_values[eva_select_id + 3])
        if eva_select_id >= 6 and eva_select_id <= 8: # eva_select_id 如果為 6 ~ 8，對應到 impossible 生出的 3 個節點
            node_values.append(eva_values[2] + eva_values[eva_select_id + 3])
    print(f'node_values: {node_values}\n')
    print('\n-----Leave Evaluator Tree-----\n')
    return max(node_values), eva_infos # 回傳最佳解(最高分的)、整棵樹的資訊

def get_prompt(task, x, y, eva_y): # 獲取提議
    eva_propose_prompt = task.evaluator_propose_prompt_wrap(x, y, eva_y) # 獲取提議提示
    eva_proposals = gpt(eva_propose_prompt, n=1, stop=None)[0].split('\n') # 使用GPT生成提議
    print(f'Propose response of evaluator: {eva_proposals}\n')
    return [eva_y + _ + '\n' for _ in eva_proposals] # 返回提議列表

def evaluator_get_value(task, x, y, eva_y, n_evaluate_sample, cache_value=True): # 根據任務、輸入 x、輸出 y 和評估樣本數量來獲得單層單一節點的值
    eva_value_prompt = task.evaluator_value_prompt_wrap(x, y, eva_y) # 生成值提示
    if cache_value and eva_value_prompt in task.value_cache:  # 如果啟用值緩存且值提示已經存在於任務的值緩存中，則從緩存中返回值
        return task.value_cache[eva_value_prompt]
    eva_value_outputs = gpt(eva_value_prompt, n=n_evaluate_sample, stop=None) # 通過 GPT 模型獲取值輸出列表
    eva_value = task.evaluator_value_outputs_unwrap(eva_y, eva_value_outputs) # 解析值輸出列表以獲取最終值
    if cache_value: # 如果啟用值緩存，則將值緩存到任務中
        task.value_cache[eva_value_prompt] = eva_value
    print(f'\nEvaluator value: {eva_value}\n')
    return eva_value

def evaluator_get_values(task, x, y, eva_ys, n_evaluate_sample, cache_value=True): # 得到該層所有節點的值，這裡的 eva_ys 原名稱為 eva_all_ys
    eva_values = [] # 儲存每個部分輸出的值
    local_value_cache = {} # 用於本地緩存的字典
    for eva_y in eva_ys:  # each partial output
        if eva_y in local_value_cache:  # avoid duplicate candidates
            eva_value = 0
        else:    
            eva_value = evaluator_get_value(task, x, y, eva_y, n_evaluate_sample, cache_value=cache_value) # 獲取值並加入到本地緩存中
            local_value_cache[y] = eva_value
        eva_values.append(eva_value) # 將值加入到結果列表中
    return eva_values